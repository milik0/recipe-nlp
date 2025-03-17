import streamlit as st
import data

st.title("📜 Recipe List")

recipe_name = st.selectbox("Choose a recipe:", list(data.recipes.keys()))

if st.button("View Recipe"):
    st.session_state["selected_recipe"] = recipe_name
    st.switch_page("app/recipe_page.py")
