import pandas as pd
import numpy as np
import optuna 
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline


import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('loan_approval_dataset.csv')
df.columns = df.columns.str.strip()
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
# print(df.head())
# print(df.columns)
# print(df.isnull().sum())
# numerical_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
#                 'cibil_score', 'residential_assets_value', 'commercial_assets_value', 
#                 'luxury_assets_value', 'bank_asset_value']
numerical_df = df.select_dtypes(include=[np.number])
numerical_cols = numerical_df.columns.drop('loan_id')
# CHECK FOR NEGATIVE VALUES IN MY DATA
# for col in numerical_cols:
#     neg_count = (df[col] < 0).sum()
#     if neg_count > 0:
#         print(f"Alert! {col} has {neg_count} negative values.")
#     else:
#         print(f"{col}: Clear (No negatives)")
        # CONVERTS THE NEGATIVE DATA IN MY FEATURE TO 0
df.loc[df['residential_assets_value'] < 0, 'residential_assets_value'] = 0
# PREPROCESSING STAGE
num_features = numerical_cols

categorical_features = ['education', 'self_employed']
# THE TRANSFORMER
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)
# SPLITTING INTO FEATUES AND TARGET
X = df.drop(['loan_id', 'loan_status'], axis=1)
y = df['loan_status'].map({'Approved': 1, 'Rejected': 0})

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print(f"Training rows: {X_train.shape[0]}")
# print(f"Testing rows: {X_test.shape[0]}")

# HYPERPARAMETER 
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }

    model = XGBClassifier(**param) 
# the ** is used to unpack a dictionary while * is used to unpack a list

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

    return score.mean()

# USING OPTUNA FOR HYPERPARAMETER TUNING
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=50)

# print("Best Parameters found by optuna: ")
# print(study.best_params)
# Extract the best hyperparameter gotten by optuna
best_params = {'n_estimators': 804, 'max_depth': 9, 'learning_rate': 0.013821963806323095, 
                'subsample': 0.9745579843338151, 'colsample_bytree': 0.8412305387389848}

final_model = XGBClassifier(**best_params,random_state=42)

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', final_model)
])

final_pipeline.fit(X_train, y_train)

test_score = final_pipeline.score(X_test, y_test)
print(f"Test Accuracy: {test_score:.4f}")

import joblib
joblib.dump(final_pipeline, 'loan_approval_model.pkl')
print("Model saved as 'loan_approval_model.pkl'")