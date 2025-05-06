import streamlit as st
import random
import time
import base64

def page_llm():
    st.title("Auguste Gusteau, The CookBro")
    st.markdown(
        "*Try our recipe generator made with Gusteau, The Renowned French Cook, with a 5-Star Michelin Restaurant. Don't forget that Anyone can cook!*"
    )

    def get_base64_image(path):
        with open(path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    GUSTEAU_IMG_PATH = "img/Gusteau.png"  # Big central image
    GUSTEAU_2_IMG_PATH = "img/Gusteau_2.png"  # Assistant's avatar
    USER_IMG_PATH = "img/ratatouille.png"

    gusteau_base64 = get_base64_image(GUSTEAU_IMG_PATH)
    gusteau_2_base64 = get_base64_image(GUSTEAU_2_IMG_PATH)
    user_base64 = get_base64_image(USER_IMG_PATH)

    # Display centered image via HTML for Gusteau
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{gusteau_base64}" width="600"/>
            <p><em>Auguste Gusteau</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

    # Display chat messages with avatars
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            with st.chat_message("assistant", avatar=GUSTEAU_2_IMG_PATH):  # Updated avatar
                st.markdown(message["content"])
        else:
            with st.chat_message("user", avatar=USER_IMG_PATH):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_IMG_PATH):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=GUSTEAU_2_IMG_PATH):  # Updated avatar
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = random.choice(
                [
                    "Hello there! How can I assist you today?",
                    "Hi, human! Is there anything I can help you with?",
                    "Do you need help?",
                ]
            )
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})
