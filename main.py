import optuna
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

import sklearn
sklearn.set_config(transform_output="pandas")

st.title('Catboost prediction')
st.divider()

# Кнопка для загрузки csv-файла
st.subheader('CSV-file uploader')
data = st.file_uploader('**Load your csv-file**', type='csv')

if data is not None:
    df = pd.read_csv(data)
    st.write(df.head(5))
else:
    df = pd.read_csv('heart.csv')
    st.write(df.head(5))

st.divider()

X, y = df.drop('HeartDisease', axis=1), df['HeartDisease']

# Подгрузим данные для обучения импутера и энкодера
df_to_learn = pd.read_csv('heart.csv')

X_l, y_l = df_to_learn.drop(
    'HeartDisease', axis=1), df_to_learn['HeartDisease']
X_train, X_valid, y_train, y_valid = train_test_split(
    X_l, y_l, test_size=0.2, stratify=y_l, random_state=1111)

my_imputer = ColumnTransformer(
    transformers=[('num_imputer', SimpleImputer(strategy='median'), ['Age'])],
    verbose_feature_names_out=False,
    remainder='passthrough'
)

filled_data = my_imputer.fit_transform(X_train)

# Cделаю кодирование и нормальизацию данных сразу в одном ColimnTransformer
ordinal_encoding_columns = ['Sex', 'RestingECG', 'ExerciseAngina']
one_hot_encoding_columns = ['ChestPainType', 'ST_Slope']
standard_scaler_columns = ['Age', 'RestingBP',
                           'Cholesterol', 'MaxHR', 'Oldpeak']
scaler_and_encoder = ColumnTransformer(
    [
        ('ordinal_encoding', OrdinalEncoder(), ordinal_encoding_columns),
        ('one_hot_encoding_columns', OneHotEncoder(
            sparse_output=False), one_hot_encoding_columns),
        ('scaling_num_columns', StandardScaler(), standard_scaler_columns)

    ],
    verbose_feature_names_out=False,
    remainder='passthrough'

)

processed_data = scaler_and_encoder.fit_transform(filled_data, y)

preprocessor = Pipeline(
    [
        ('imputer', my_imputer),
        ('scaler_and_encoder', scaler_and_encoder)
    ]
)

prepr_x_train = preprocessor.fit_transform(X_train)
prepr_x_valid = preprocessor.transform(X_valid)

# Экспортирую обученную модель CatBoost
loaded_model = CatBoostClassifier()
loaded_model.load_model("catboost_model.cbm")

st.subheader('CatBoost Accuracy')
st.metric('Accuracy', round(accuracy_score(
    y_valid, loaded_model.predict(prepr_x_valid)), 4))

st.divider()

st.subheader('Encoded and scaled data')
preprocessed_x_to_pred = preprocessor.transform(X)
st.write(preprocessed_x_to_pred)

st.divider()

st.subheader('Predicted Data')
x_pred = loaded_model.predict(preprocessed_x_to_pred)
x_pred_df = pd.DataFrame(x_pred, columns=['HeartDisease predicted'])
st.dataframe(x_pred_df, hide_index=True)
