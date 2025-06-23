import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

df = pd.read_csv('loan_data.csv')

loan_default_map = {'N': 0, 'Y': 1}
education_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3}
home_ownership_map = {'OWN': 0, 'MORTGAGE': 1, 'RENT': 2, 'OTHER': 3}

class MapEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mapping_dict, fill_value=-1):
        self.mapping_dict = mapping_dict
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col, mapping in self.mapping_dict.items():
            X_copy[col] = X_copy[col].map(mapping).fillna(self.fill_value)
        return X_copy

one_hot_features = ['person_gender', 'loan_intent']
map_features = ['previous_loan_defaults_on_file', 'person_education', 'person_home_ownership']
skewed_features = ['person_age', 'person_income', 'person_emp_exp', 'loan_percent_income', 'cb_person_cred_hist_length']
non_skewed_features = ['loan_amnt', 'loan_int_rate', 'credit_score']

all_maps = {
    'previous_loan_defaults_on_file': loan_default_map,
    'person_education': education_map,
    'person_home_ownership': home_ownership_map
}

log_transformer = FunctionTransformer(np.log1p, validate=True)
robust_scaler = RobustScaler()

preprocessor = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False), one_hot_features),
    ('map', MapEncoder(all_maps), map_features),
    ('skewed_num', Pipeline([('log', log_transformer), ('scale', robust_scaler)]), skewed_features),
    ('non_skewed_num', Pipeline([('scale', robust_scaler)]), non_skewed_features)
])

X = df.drop(columns='loan_status')
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

catboost_params = {
    'depth': 8,
    'iterations': 200,
    'l2_leaf_reg': 3,
    'learning_rate': 0.1,
    'verbose': 0,
    'random_seed': 42,
    'early_stopping_rounds': 20
}

model = CatBoostClassifier(**catboost_params)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

pickle_model_path = 'model.pkl'
with open(pickle_model_path, 'wb') as f:
    pickle.dump(pipeline, f)
