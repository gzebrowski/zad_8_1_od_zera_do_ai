import pandas as pd
import streamlit as st
from pycaret.regression import load_model, predict_model


@st.cache_resource
def get_dataframe(filename):
    _df = pd.read_csv(filename)
    _df.set_index(['country_name'], inplace=True)
    return _df


@st.cache_data
def get_data(_df):
    fields: list[str] = [c for c in _df.columns if c not in ['year', 'country_name', 'happiness_score']]
    aggr_data = _df[fields].agg(['mean', 'max', 'min']).to_dict()
    return fields, aggr_data


@st.cache_resource
def cache_load_regression_model():
    return load_model('world_happines_regression_pipeline')


df = get_dataframe('world-happiness-report_2023.csv')
fields, aggr_data = get_data(df)
world_happines_model = cache_load_regression_model()


with st.sidebar:
    user_values = {}
    for field in fields:
        user_values[field] = st.slider(field.replace('_', ' ').capitalize(), min_value=aggr_data[field]['min'],
                                       max_value=aggr_data[field]['max'], value=aggr_data[field]['mean'])

ret_val = predict_model(world_happines_model, data=pd.DataFrame([user_values]))
predicted_value = ret_val.prediction_label[0]

st.write('Wynik:', predicted_value)

col1, col2 = st.columns(2)
with col1:
    st.write('Kraje mające niższy happiness_score')
    st.dataframe(df[df['happiness_score'] < predicted_value][['happiness_score']].sort_values(
        'happiness_score', ascending=False), use_container_width=True)

with col2:
    st.write('Kraje mające wyższy lub równy happiness_score')
    st.dataframe(df[df['happiness_score'] >= predicted_value][['happiness_score']].sort_values(
        'happiness_score', ascending=True), use_container_width=True)
