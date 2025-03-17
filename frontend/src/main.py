import streamlit as st
import pandas as pd

# Sample data
data = pd.DataFrame({
    "alim_nom_eng": ["Apple", "Banana", "Orange"],
    "recipe": [
        "Apple Pie: Mix apples, sugar, and cinnamon. Bake at 180°C for 40 min.",
        "Banana Smoothie: Blend banana, milk, and honey until smooth.",
        "Orange Juice: Squeeze fresh oranges and serve chilled."
    ]
})

# Title
st.title("📜 Recipe Collection")

# Initialize session state
if "selected_ingredients" not in st.session_state:
    st.session_state.selected_ingredients = []
if "show_recipe" not in st.session_state:
    st.session_state.show_recipe = False

# Multi-select dropdown for choosing ingredients
selected_ingredients = st.multiselect(
    "Pick ingredients:", data["alim_nom_eng"], default=st.session_state.selected_ingredients
)

# Update session state if ingredients change
if selected_ingredients != st.session_state.selected_ingredients:
    st.session_state.selected_ingredients = selected_ingredients
    st.session_state.show_recipe = False

# Display combined recipe using all selected ingredients
if selected_ingredients:
    if st.button("Generate Recipe", key="generate"):
        st.session_state.show_recipe = True
    
    if st.session_state.show_recipe:
        st.header("🧑🍳 Combined Recipe")
        st.subheader(f"🍽️ Recipe using {', '.join(selected_ingredients)}")
        
        if len(selected_ingredients) == 1:
            # Show original recipe for single ingredient
            recipe_text = data.loc[data["alim_nom_eng"] == selected_ingredients[0], "recipe"].values[0]
            st.write(recipe_text)
        else:
            # For multiple ingredients, create a combined recipe
            if all(fruit in selected_ingredients for fruit in ["Apple", "Banana", "Orange"]):
                st.write("**Triple Fruit Salad**")
                st.write("""
                1. Dice 1 apple into small cubes
                2. Slice 1 banana into rounds
                3. Segment 1 orange, removing all pith
                4. Combine all fruits in a large bowl
                5. Add 2 tbsp of honey and 1 tbsp of lemon juice
                6. Toss gently and chill for 30 minutes
                7. Optional: sprinkle with cinnamon before serving
                """)
            elif "Apple" in selected_ingredients and "Banana" in selected_ingredients:
                st.write("**Apple Banana Smoothie Bowl**")
                st.write("""
                1. Peel and slice 1 banana and 1 apple
                2. Blend with 1/2 cup of yogurt and 1/4 cup milk
                3. Pour into a bowl
                4. Top with apple slices and a drizzle of honey
                5. Optional: add granola or nuts for crunch
                """)
            elif "Banana" in selected_ingredients and "Orange" in selected_ingredients:
                st.write("**Tropical Sunrise Smoothie**")
                st.write("""
                1. Peel 1 banana and 1 orange
                2. Blend with 1/2 cup of ice and 1/4 cup of coconut water
                3. Add a teaspoon of honey if desired
                4. Garnish with an orange slice
                """)
            elif "Apple" in selected_ingredients and "Orange" in selected_ingredients:
                st.write("**Citrus Apple Salad**")
                st.write("""
                1. Dice 1 apple into small cubes
                2. Segment 1 orange, removing all pith
                3. Mix together in a bowl
                4. Add a dressing of 1 tbsp honey and 1 tsp lemon juice
                5. Chill for 20 minutes before serving
                """)
            else:
                # Generic response for any other combination
                ingredients_list = ", ".join(selected_ingredients)
                st.write(f"**{ingredients_list} Medley**")
                st.write(f"""
                1. Prepare all ingredients: wash and cut {ingredients_list} into bite-sized pieces
                2. Combine in a large bowl
                3. Add a simple dressing of honey and lemon juice
                4. Mix gently and serve chilled
                """)

st.write("Bon appétit! 🍽️")
