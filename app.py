import pandas as pd
import streamlit as st
from pycaret.regression import predict_model

from app_helpers import slugify
from pycaret_helper import prepare_regression_model

df = pd.read_csv('world-happiness-report.csv')

columns = df.columns
new_columns = [slugify(c, '_') for c in columns]
df.columns = new_columns

descriptions = dict(list(zip(new_columns, columns)))

df2 = df.copy()[df['year'] == 2023][[c for c in new_columns if c not in ['year']]]
df2.set_index('country_name')

world_happines_model = prepare_regression_model(df2, 'world_happines_regression_pipeline', 'happiness_score')


fields = [c for c in new_columns if c not in ['year', 'country_name', 'happiness_score']]
means = df2[fields].mean().to_dict()
mins = df2[fields].min().to_dict()
maxes = df2[fields].max().to_dict()

aggr_data = df2[fields].agg(['mean', 'max', 'min']).to_dict()

with st.sidebar:
    user_values = {}
    for field in fields:
        user_values[field] = st.slider(descriptions[field], min_value=aggr_data[field]['min'],
                                       max_value=aggr_data[field]['max'], value=aggr_data[field]['mean'])

user_values['country_name'] = ''
# hp_p = HapinesParam(**user_values)
ret_val = predict_model(world_happines_model, data=pd.DataFrame([user_values]))
predicted_value = ret_val.prediction_label[0]

st.write('Wynik:', predicted_value)

col1, col2 = st.columns(2)
with col1:
    st.write('Kraje mające niższy happiness_score')
    st.write(df2[df2['happiness_score'] < predicted_value][['country_name', 'happiness_score']].sort_values(
        'happiness_score', ascending=False))

with col2:
    st.write('Kraje mające wyższy lub równy happiness_score')
    st.write(df2[df2['happiness_score'] >= predicted_value][['country_name', 'happiness_score']].sort_values(
        'happiness_score', ascending=True))
