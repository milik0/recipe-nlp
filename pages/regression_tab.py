from pages.regression.train_models import select_model, create_feature_matrices, train_test_split_custom
import streamlit as st
import pandas as pd
from nltk.stem import PorterStemmer
import os
from pages.regression.preprocessing import process_data

ps = PorterStemmer()

models_dict = {
    "Linear Regression - TF-IDF": "lr_tf",
    "Bayesian Ridge - TF-IDF": "bayesian_tf",
    "Feedforward Neural Network - TF-IDF": "forward_nn_tf",
    "Gated Recurrent Unit - TF-IDF": "gru_tf",
    "Linear Regression - Word2Vec": "lr_w2v",
    "Bayesian Ridge - Word2Vec": "bayesian_w2v",
    "Feedforward Neural Network - Word2Vec": "forward_nn_w2v",
    "Gated Recurrent Unit - Word2Vec": "gru_w2v",
}


def page_regression():
    st.title("⏱️ Time Cooking Predictor")
    st.markdown("*Never lose track of time with our Time Cooking Predictor !*")

    # Sidebar model selection
    st.sidebar.markdown("### 📌 Models")
    selected_models = {}
    for model_name in models_dict.keys():
        selected_models[model_name] = st.sidebar.checkbox(f"{model_name}")

    selected_model_names = [
        name for name, selected in selected_models.items() if selected
    ]

    if not selected_model_names:
        st.sidebar.warning("Please select at least one model for comparison")
        return

    # Compute button
    if st.button("🚀 Compute Predictions"):
        data_path = 'data/processed/preprocessed_recipe.csv'
        
        message_placeholder = st.empty()  # create a placeholder

        if os.path.isfile(data_path):
            message_placeholder.info("Loading preprocessed data...")
            data = pd.read_csv(data_path)
            message_placeholder.empty()  # clear the message
        else:
            message_placeholder.warning("Preprocessed file not found. Running preprocessing...")
            data = process_data(save=True)
            message_placeholder.success("Preprocessing done. Loading data...")
            st.sleep(1)
            message_placeholder.empty()
        
        feature_matrices_tf = create_feature_matrices(data, 'tf')
        feature_matrices_w2v = create_feature_matrices(data, 'w2v')
        
        X_train_tf, X_test_tf, y_train_tf, y_test_tf, _, _ = train_test_split_custom(data, feature_matrices_tf)
        X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v, _, _ = train_test_split_custom(data, feature_matrices_w2v)

        df_results = pd.DataFrame()
        for model_name in selected_model_names:
            with st.spinner(f"Calculating {model_name} model..."):
                model_key = models_dict[model_name]
                if model_key.endswith('tf'):
                    results = select_model(model_key, X_train_tf, X_test_tf, y_train_tf, y_test_tf)
                if model_key.endswith('w2v'):
                    results = select_model(model_key, X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v)
                else:
                    return #TODO
                
                results["Model"] = model_name
                df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index=True)

        column_order = ["Model", "MAE", "MSE", "RMSE", "R2"]
        df_results = df_results[[col for col in column_order if col in df_results.columns]]

        st.subheader("Model Performance Metrics")
        st.dataframe(df_results.style.format(precision=4))

