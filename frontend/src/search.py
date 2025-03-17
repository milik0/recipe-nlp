import streamlit as st
import pandas as pd
ingredients = ['Apple', 'Banana', 'Orange', '']

data = pd.read_excel("./healthy.xls")
print(data['alim_nom_eng'])

st.checkbox('Yes')
st.button('Click Me')
st.selectbox('Pick a fruit', data['alim_nom_eng'])
st.multiselect('Choose a planet', ['Jupiter', 'Mars', 'Neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0, 50)
