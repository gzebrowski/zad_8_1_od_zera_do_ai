import pandas as pd
import streamlit as st
from pycaret.regression import predict_model

from app_helpers import slugify
from pycaret_helper import prepare_regression_model


@st.cache_resource
def get_data(filename):
    return pd.read_csv(filename)


@st.cache_resource
def cache_load_regression_model(_df):
    dt2 = prepare_regression_model(_df, 'world_happines_regression_pipeline', 'happiness_score',
                                   ignore_features=['country_name'])
    return dt2


df = get_data('world-happiness-report.csv')

columns = df.columns.to_list()
new_columns = list(map(lambda c: slugify(c, '_'), columns))
df2 = df.copy()
df2.columns = new_columns

descriptions = dict(list(zip(new_columns, columns)))

df3 = df2.copy()[df2['year'] == 2023][[c for c in new_columns if c not in ['year']]]
df3.set_index(['country_name'], inplace=True)

world_happines_model = cache_load_regression_model(df3)

fields = [c for c in new_columns if c not in ['year', 'country_name', 'happiness_score']]
means = df3[fields].mean().to_dict()
mins = df3[fields].min().to_dict()
maxes = df3[fields].max().to_dict()

aggr_data = df3[fields].agg(['mean', 'max', 'min']).to_dict()

with st.sidebar:
    user_values = {}
    for field in fields:
        user_values[field] = st.slider(descriptions[field], min_value=aggr_data[field]['min'],
                                       max_value=aggr_data[field]['max'], value=aggr_data[field]['mean'])

ret_val = predict_model(world_happines_model, data=pd.DataFrame([user_values]))
predicted_value = ret_val.prediction_label[0]

st.write('Wynik:', predicted_value)

col1, col2 = st.columns(2)
with col1:
    st.write('Kraje mające niższy happiness_score')
    st.dataframe(df3[df3['happiness_score'] < predicted_value][['happiness_score']].sort_values(
        'happiness_score', ascending=False), use_container_width=True)

with col2:
    st.write('Kraje mające wyższy lub równy happiness_score')
    st.dataframe(df3[df3['happiness_score'] >= predicted_value][['happiness_score']].sort_values(
        'happiness_score', ascending=True), use_container_width=True)
