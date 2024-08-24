import streamlit as st
import pandas as pd
import joblib
from src.pipeline.prediction_pipeline import PredictionPipeline
import numpy as np

model_path = 'artifacts/model_trainer/model.pkl'
preprocessor_path = 'artifacts/data_transformation/preprocessor.pkl'

try:
    loaded_model = joblib.load(model_path)
    loaded_preprocessor = joblib.load(preprocessor_path)
    print(f"Model loaded successfully from {model_path}")
    print(f'Preprocessor loaded successfully from {preprocessor_path}')
except FileNotFoundError:
    print(f"Error: Model file '{model_path}' not found.")
    loaded_model = None
    loaded_preprocessor = None
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None
    loaded_preprocessor = None

def predictions(Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity):
    if loaded_model is None or loaded_preprocessor is None:
        return "Model or Preprocessor not loaded"
    
    custom_data = {
        'Size': [Size],
        'Weight': [Weight],
        'Sweetness': [Sweetness],
        'Crunchiness': [Crunchiness],
        'Juiciness': [Juiciness],
        'Ripeness': [Ripeness],
        'Acidity': [Acidity]
    }

    input_df = pd.DataFrame(custom_data)

    # Preprocess the data
    input_df_transformed = loaded_preprocessor.transform(input_df)

    # Make a prediction
    prediction = loaded_model.predict(input_df_transformed)

    return prediction[0]  # Assuming the model returns an array, take the first element

def main():
    st.title('Apple Quality Checker')
    st.header("Please enter the data")

    Size = st.number_input("Size", min_value=-7.0, max_value=6.0, step=0.01)
    Weight = st.number_input("Weight", min_value=-7.0, max_value=5.79, step=0.01)
    Sweetness = st.number_input("Sweetness", min_value=-6.89, max_value=6.37, step=0.01)
    Crunchiness = st.number_input("Crunchiness", min_value=-6.0, max_value=7.6, step=0.01)
    Juiciness = st.number_input("Juiciness", min_value=-5.9, max_value=7.3, step=0.01)
    Ripeness = st.number_input("Ripeness", min_value=-5.86, max_value=7.27, step=0.01)
    Acidity = st.number_input("Acidity", min_value=-7.0, max_value=7.4, step=0.01)


    if st.button('Predict'):
        try:
            result = predictions(Size, Weight, Sweetness, Crunchiness, Juiciness, Ripeness, Acidity)
            st.write(f"Raw Prediction Result: {result}")

            # Assuming result is a string label or binary classification
            if result == 'good':
                st.success("Apple is Good")
            elif result == 'bad':
                st.error("Apple is Bad")
            else:
                st.write("Error: Unexpected result format.")
        except Exception as e:
            st.write(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
