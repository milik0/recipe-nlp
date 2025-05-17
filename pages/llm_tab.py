import streamlit as st
import random
import time
import base64
import tensorflow as tf
import numpy as np
import pickle
import os
import re
import sys
from tensorflow.keras.preprocessing.sequence import pad_sequences

sys.path.insert(1, "../..")

from src.generator.models.optimized_feedforward_nn import OptimizedBatchFeedForwardNN
from src.generator.models.ngram import RecipeNGramModel

# Set GPU to not be used
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Model utility functions
def format_ingredients_for_model(ingredients):
    """Format ingredients as expected by the model"""
    if isinstance(ingredients, str):
        # If comma-separated string, split into list
        ingredients_list = [ing.strip() for ing in ingredients.split(",")]
    else:
        # Already a list
        ingredients_list = ingredients

    # Format as quoted items in brackets
    return "[" + ", ".join(f"'{ing}'" for ing in ingredients_list) + "]"


def generate_text(model, tokenizer, seed_text, max_length=100, temperature=1.0):
    """Generate recipe text based on ingredients"""
    MAX_SEQUENCE_LENGTH = 100
    # Convert seed text to tokens
    input_tokens = tokenizer.texts_to_sequences([seed_text])[0]

    # Initialize text generation
    generated_text = seed_text

    for i in range(max_length):
        # Truncate input if too long
        if len(input_tokens) > MAX_SEQUENCE_LENGTH - 1:
            input_tokens = input_tokens[-(MAX_SEQUENCE_LENGTH - 1) :]

        # Pad the input to the required length
        padded_tokens = pad_sequences(
            [input_tokens], maxlen=MAX_SEQUENCE_LENGTH - 1, padding="pre"
        )

        # Create input tensor
        input_tensor = tf.convert_to_tensor(padded_tokens, dtype=tf.int32)

        # Generate predictions
        predictions = model(input_tensor)

        # Apply temperature sampling
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()

        # Add predicted token to input
        input_tokens.append(predicted_id)

        # Get the corresponding word (ensure it exists in vocabulary)
        if predicted_id < len(tokenizer.index_word):
            predicted_word = tokenizer.index_word.get(predicted_id, "")
            generated_text += " " + predicted_word

            # Stop if EOS token is generated
            if predicted_word == "<EOS>":
                break
        else:
            generated_text += " <UNKNOWN>"

    return generated_text


class GruRNNModel:
    def __init__(self):
        super().__init__()
        self.name = "RNN (GRU)"
        self.model = None
        self.tokenizer = None

    def load_model(self, model_path, tokenizer_path):
        with open(tokenizer_path, "rb") as handle:
            self.tokenizer = pickle.load(handle)

        self.model = tf.keras.models.load_model(model_path)

    def predict(self, input_text, max_length=200, temperature=1.0):
        """Generate recipe from ingredients"""
        if not self.model or not self.tokenizer:
            return "Model not loaded properly."

        formatted_ingredients = format_ingredients_for_model(input_text)
        seed_text = f"Ingredients: {formatted_ingredients} | Recipe Steps: "

        # Generate recipe
        generated_text = generate_text(
            self.model,
            self.tokenizer,
            seed_text,
            max_length=max_length,
            temperature=temperature,
        )

        # Clean up the output by removing the seed text
        result = generated_text.replace(seed_text, "")

        # Remove any EOS tokens
        result = result.replace("<EOS>", "").strip()

        return result


# Function to format recipe output nicely
def format_recipe_output(generated_text):
    """Format the generated recipe text nicely for display"""
    # Split the generated text to get just the recipe part
    if "Recipe Steps:" in generated_text:
        recipe_part = generated_text.split("Recipe Steps:")[1].strip()
    else:
        recipe_part = generated_text.strip()

    # Remove any EOS tokens or unknown tokens
    recipe_part = recipe_part.replace("<EOS>", "").replace("<UNKNOWN>", "").strip()

    # Try to format as numbered steps
    steps = []
    current_step = ""

    for part in recipe_part.split("."):
        if part.strip():
            current_step += part.strip() + "."
            if len(current_step) > 20:  # Reasonable step length
                steps.append(current_step.strip())
                current_step = ""

    # Add any remaining step
    if current_step.strip():
        steps.append(current_step.strip())

    # Format with numbers
    formatted_recipe = ""
    for i, step in enumerate(steps, 1):
        formatted_recipe += f"{i}. {step}\n\n"

    return formatted_recipe


# Function to extract ingredients from user message
def extract_ingredients(message):
    """Extract ingredients from user message"""
    ingredients = ""

    # Pattern matching for various ways users might provide ingredients
    if "ingredients:" in message.lower():
        ingredients = message.lower().split("ingredients:")[1].strip()
    elif "using" in message.lower() and any(
        word in message.lower() for word in ["ingredients", "items", "foods"]
    ):
        ingredients_part = message.lower().split("using")[1]
        # Find the part after "ingredients", "items", or "foods"
        for term in ["ingredients", "items", "foods"]:
            if term in ingredients_part:
                ingredients = ingredients_part.split(term)[1].strip(" :.,")
                break
        if not ingredients:  # If no specific term found
            ingredients = ingredients_part.strip(" :.,")
    elif re.search(r"recipe\s+with\s+([\w\s,]+)", message.lower()):
        # Match "recipe with X, Y, Z"
        match = re.search(r"recipe\s+with\s+([\w\s,]+)", message.lower())
        if match:
            ingredients = match.group(1).strip()
    elif re.search(r"make.*using\s+([\w\s,]+)", message.lower()):
        # Match "make something using X, Y, Z"
        match = re.search(r"make.*using\s+([\w\s,]+)", message.lower())
        if match:
            ingredients = match.group(1).strip()
    elif re.search(r"cook.*with\s+([\w\s,]+)", message.lower()):
        # Match "cook something with X, Y, Z"
        match = re.search(r"cook.*with\s+([\w\s,]+)", message.lower())
        if match:
            ingredients = match.group(1).strip()
    elif "i have" in message.lower() and any(
        word in message.lower() for word in ["recipe", "cook", "make", "dish"]
    ):
        # Match "I have X, Y, Z. What can I make?"
        ingredients_part = message.lower().split("i have")[1].strip()
        # Find where the question starts
        for question_start in ["what", "can", "how", "?"]:
            if question_start in ingredients_part:
                ingredients = ingredients_part.split(question_start)[0].strip(" :.,")
                break
        if not ingredients:  # If no question found
            ingredients = ingredients_part.strip(" :.,")
    elif re.search(r"i have\s+([\w\s,]+)", message.lower()):
        # Simple "I have X, Y, Z"
        match = re.search(r"i have\s+([\w\s,]+)", message.lower())
        if match:
            ingredients = match.group(1).strip(" :.,?")
    elif any(
        message.lower().startswith(term)
        for term in ["can you", "could you", "please", "help"]
    ):
        # Check if the message contains a comma-separated list anywhere
        if "," in message:
            # Extract the part of the message with commas
            comma_parts = [part for part in re.split(r"[.?!]", message) if "," in part]
            if comma_parts:
                ingredients = comma_parts[0].strip(" :.,?")

    # If still nothing found, check if the message itself looks like a comma-separated list
    if not ingredients and "," in message and len(message.split(",")) >= 2:
        # The message might just be a list of ingredients
        ingredients = message.strip()

    return ingredients


def page_llm():
    st.title("Auguste Gusteau, The CookBro")
    st.markdown(
        "*Try our recipe generator made with Gusteau, The Renowned French Cook, with a 5-Star Michelin Restaurant. Don't forget that Anyone can cook!*"
    )

    # Create a model factory that loads different models
    @st.cache_resource
    def load_models():
        models = {}

        # Load RNN model
        try:
            rnn_model = GruRNNModel()
            rnn_model.load_model(
                model_path="./models/saved/generator/recipe_rnn_model.keras",
                tokenizer_path="./models/saved/generator/recipe_tokenizer.pickle",
            )
            models["rnn"] = rnn_model
        except Exception as e:
            st.error(f"Error loading RNN model: {e}")

        # Load FFNN model
        try:
            ffnn_model = OptimizedBatchFeedForwardNN()
            ffnn_model.load("./models/saved/generator/ffnn.keras")
            models["ffnn"] = ffnn_model
        except Exception as e:
            st.error(f"Error loading FFNN model: {e}")

        # Load NGram model
        try:
            ngram_model = RecipeNGramModel()
            ngram_model.load(
                "./models/saved/generator/ngram.pkl",
            )
            models["ngram"] = ngram_model
        except Exception as e:
            st.error(f"Error loading NGram model: {e}")

        return models

    try:
        all_models = load_models()
        model_loaded = len(all_models) > 0

        # Set default model if available
        if "rnn" in all_models:
            default_model = "rnn"
        elif "ffnn" in all_models:
            default_model = "ffnn"
        elif "ngram" in all_models:
            default_model = "ngram"
        else:
            default_model = None
            model_loaded = False

    except Exception as e:
        st.error(f"Error initializing models: {e}")
        all_models = {}
        model_loaded = False
        default_model = None

    def get_base64_image(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    GUSTEAU_IMG_PATH = "img/Gusteau.png"  # Big central image
    GUSTEAU_2_IMG_PATH = "img/Gusteau_2.png"  # Assistant's avatar
    USER_IMG_PATH = "img/ratatouille.png"

    # Display centered image via HTML for Gusteau
    try:
        gusteau_base64 = get_base64_image(GUSTEAU_IMG_PATH)
        st.markdown(
            f"""
            <div style="text-align: center;">
                <img src="data:image/png;base64,{gusteau_base64}" width="600"/>
                <p><em>Auguste Gusteau</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.warning(f"Could not load main image: {e}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Bonjour! I am Auguste Gusteau. Tell me what ingredients you have directly in the chat, and I'll create a wonderful recipe for you! You can also use the form below. Anyone can cook with my help!",
            }
        ]

    # Add recipe generation UI
    st.subheader("Generate a Recipe")

    with st.form("recipe_form"):
        # Add model selector
        model_options = {
            "rnn": "RNN (GRU) - Most creative",
            "ffnn": "Feed-Forward NN - More structured",
            "ngram": "N-Gram - Closest to original recipes",
        }

        available_models = {k: v for k, v in model_options.items() if k in all_models}

        if available_models:
            selected_model = st.selectbox(
                "Select Recipe Generation Model:",
                options=list(available_models.keys()),
                format_func=lambda x: available_models.get(x, x),
                index=(
                    list(available_models.keys()).index(default_model)
                    if default_model in available_models
                    else 0
                ),
            )
        else:
            selected_model = None
            st.warning("No models available. Please check the model paths.")

        ingredients_input = st.text_input(
            "Enter your ingredients (separated by commas):",
            placeholder="chicken, butter, salt, pepper, garlic",
        )

        col1, col2 = st.columns(2)
        with col1:
            max_length = st.slider(
                "Recipe Length", min_value=50, max_value=500, value=200, step=10
            )
        with col2:
            temperature = st.slider(
                "Creativity Level", min_value=0.5, max_value=1.5, value=1.0, step=0.1
            )

        generate_button = st.form_submit_button("Generate Recipe")

    # Store recipe generation settings in session state
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = selected_model
    else:
        st.session_state.selected_model = selected_model

    if "max_length" not in st.session_state:
        st.session_state.max_length = max_length
    else:
        st.session_state.max_length = max_length

    if "temperature" not in st.session_state:
        st.session_state.temperature = temperature
    else:
        st.session_state.temperature = temperature

    # Generate recipe when button is clicked
    if generate_button and ingredients_input and model_loaded and selected_model:
        with st.spinner(
            f"Chef Gusteau is creating your recipe using the {all_models[selected_model].name} model..."
        ):
            try:
                # Generate recipe with the selected model
                if selected_model == "ffnn":
                    ingredients_input = [ingredients_input]
                print(ingredients_input)
                raw_recipe = all_models[selected_model].predict(
                    input_text=ingredients_input,
                    max_length=max_length,
                    temperature=temperature,
                )
                if selected_model == "ffnn":
                    raw_recipe = " ".join(raw_recipe[0])
                print(raw_recipe)

                # Format recipe nicely
                formatted_recipe = format_recipe_output(raw_recipe)

                # Add to chat history
                user_message = f"Please create a recipe using these ingredients: {ingredients_input}"
                st.session_state.messages.append(
                    {"role": "user", "content": user_message}
                )

                # Create assistant response with a nice header
                assistant_response = f"""
## A Recipe by Chef Gusteau
**Using {all_models[selected_model].name} model**

**Ingredients:** {ingredients_input}

### Instructions:
{formatted_recipe}

*Bon appétit! Anyone can cook!*
"""
                st.session_state.messages.append(
                    {"role": "assistant", "content": assistant_response}
                )

            except Exception as e:
                st.error(f"Error generating recipe: {e}")

    # Display chat messages with avatars
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=GUSTEAU_2_IMG_PATH):
                st.markdown(message["content"])
        else:
            with st.chat_message("user", avatar=USER_IMG_PATH):
                st.markdown(message["content"])

    # Accept user input for regular chat
    if prompt := st.chat_input("Ask Chef Gusteau anything or provide ingredients..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_IMG_PATH):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=GUSTEAU_2_IMG_PATH):
            message_placeholder = st.empty()
            full_response = ""

            # Check if this might be a request for a recipe by looking for ingredients
            ingredients = prompt

            # Check if the message is explicitly asking for a recipe
            is_recipe_request = (
                any(
                    word in prompt.lower()
                    for word in ["recipe", "make", "cook", "dish", "prepare", "create"]
                )
                or len(ingredients) > 5
            )

            if (
                ingredients
                and (is_recipe_request or "," in ingredients)
                and model_loaded
                and selected_model
            ):
                try:
                    # Let the user know we're generating a recipe
                    processing_message = f"Ah! I see you've provided ingredients. Let me create a wonderful recipe for you using my {all_models[selected_model].name} model..."
                    for chunk in processing_message.split():
                        full_response += chunk + " "
                        time.sleep(0.03)
                        message_placeholder.markdown(full_response + "▌")

                    # Generate recipe
                    with st.spinner(
                        f"Chef Gusteau is creating your recipe using the {all_models[selected_model].name} model..."
                    ):
                        if selected_model == "ffnn":
                            ingredients_input = [ingredients_input]
                        raw_recipe = all_models[selected_model].predict(
                            input_text=ingredients_input,
                            max_length=st.session_state.max_length,
                            temperature=st.session_state.temperature,
                        )
                        if selected_model == "ffnn":
                            raw_recipe = " ".join(raw_recipe)

                        # Format recipe nicely
                        formatted_recipe = format_recipe_output(raw_recipe)

                    # Create assistant response with a nice header
                    assistant_response = f"""
## A Recipe by Chef Gusteau
**Using {all_models[selected_model].name} model**

**Ingredients:** {ingredients}

### Instructions:
{formatted_recipe}

*Bon appétit! Anyone can cook!*
"""
                    # Display the recipe
                    message_placeholder.markdown(assistant_response)

                except Exception as e:
                    error_msg = f"Oh non! I had trouble creating a recipe with those ingredients: {str(e)}"
                    message_placeholder.markdown(error_msg)
                    assistant_response = error_msg
            else:
                # Handle general cooking questions or conversation
                # Check if it might be about ingredients but unclear
                if (
                    any(
                        word in prompt.lower()
                        for word in ["ingredients", "have", "got", "using"]
                    )
                    and not is_recipe_request
                ):
                    assistant_response = "It sounds like you're telling me about ingredients. If you'd like me to create a recipe, please list your ingredients separated by commas, or clearly ask for a recipe with those ingredients!"
                elif any(
                    word in prompt.lower()
                    for word in ["model", "select", "switch", "change", "ai", "neural"]
                ):
                    # User might be asking about models
                    available_models_str = ", ".join(
                        [f"{m.name}" for m in all_models.values()]
                    )
                    assistant_response = f"I can generate recipes using different AI models! Currently, I have {available_models_str} available. You can select the model in the dropdown menu above the ingredient input field."
                else:
                    # General cooking responses
                    cooking_responses = [
                        "Anyone can cook, but only the fearless can be great!",
                        "The only thing predictable about life is its unpredictability!",
                        "Great cooking is not for the faint of heart, mon ami!",
                        "You must not let anyone define your limits because of where you come from.",
                        "Cooking is like life - you never know what flavors await unless you try!",
                        "In cooking, as in life, you must be bold and adventurous!",
                        "Your ingredients may be simple, but your dish can be extraordinary!",
                        "A recipe is only as good as the chef who interprets it.",
                        "In my kitchen, there are no mistakes, only new discoveries!",
                        "The difference between a good cook and a great one is imagination!",
                    ]

                    # Check if it's a greeting
                    if any(
                        word in prompt.lower()
                        for word in ["hello", "hi", "hey", "bonjour"]
                    ):
                        assistant_response = "Bonjour! I am Auguste Gusteau. Tell me what ingredients you have, and I'll create a wonderful recipe for you! Just list your ingredients separated by commas."
                    # Check if asking for help
                    elif any(
                        word in prompt.lower()
                        for word in ["help", "how", "what can you"]
                    ):
                        assistant_response = "I can help you create delicious recipes! Simply tell me what ingredients you have by listing them separated by commas in the chat, and I'll craft a recipe for you. You can also use the form above to select which AI model to use, adjust recipe length and creativity level."
                    else:
                        assistant_response = random.choice(cooking_responses)
                        # Add a note about recipe generation
                        assistant_response += "\n\nIf you want me to create a recipe, please list your ingredients separated by commas in the chat, or use the form above to select a model and other settings!"

                # Display with typing animation
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.03)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                assistant_response = full_response

            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )


# This can be used to add to your existing Streamlit app with multiple pages
if __name__ == "__main__":
    page_llm()
