import streamlit as st
from pages.llm_tab import *
from pages.regression_tab import *

def hide_streamlit_style():
    """Used to hide the Streamlit menu and footer."""
    hide_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_style, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="'Los Pollos Hermanos' de Gusteau", page_icon="👨‍🍳"
    )
    hide_streamlit_style()

    # Sidebar for page selection
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a Page",
        (
            'Regression',
            'Auguste Gusteau, The CookBro'
        ),
    )

    if page == 'Regression':
        page_regression()
    
    if page == "Auguste Gusteau, The CookBro":
        page_llm()


if __name__ == "__main__":
    main()
