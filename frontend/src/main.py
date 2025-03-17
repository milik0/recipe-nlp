import streamlit as st
import pandas as pd
from data import recipes

# Title and description of the app
st.title("🍳 Recipe Finder Dashboard")
st.write(
    "Search for your favorite recipes based on available ingredients, cuisine, or recipe type."
)

# Filter button
if st.button("Search Recipes"):
    filtered_recipes = []

    for name, recipe in recipes.items():
        # Check if ingredients match the selected ingredients
        if all(ingredient in available_ingredients for ingredient in recipe["ingredients"]):
            if recipe_type == "All" or recipe["type"] == recipe_type:
                filtered_recipes.append((name, recipe))

    if filtered_recipes:
        st.subheader(f"Found {len(filtered_recipes)} recipe(s)")
        # Display filtered recipes in a clickable list
        for name, recipe in filtered_recipes:
            if st.button(f"View {name}"):
                st.session_state.selected_recipe = name
                st.session_state.recipe_details = recipe
                st.experimental_rerun()  # Redirect to details page
    else:
        st.warning("No recipes found based on the selected criteria.")