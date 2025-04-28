import streamlit as st
import data

st.title("🍽 Recipe Details")

recipe_name = st.session_state.get("selected_recipe", None)

if recipe_name and recipe_name in data.recipes:
    recipe = data.recipes[recipe_name]

    st.header(recipe_name)
    st.subheader("Ingredients:")
    st.write("\n".join(f"- {item}" for item in recipe["ingredients"]))

    st.subheader("Instructions:")
    st.write(recipe["instructions"])
else:
    st.warning("No recipe selected. Go back to the main page.")

if st.button("🔙 Back to List"):
    st.switch_page("app/main.py")
