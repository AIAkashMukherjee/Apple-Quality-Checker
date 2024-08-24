import os
import sys
import pandas as pd
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
from src.utlis.utlis import load_obj
import numpy as np

class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        """
        Load the preprocessor and model, then make predictions on the provided features.
        """
        try:
            # Paths to the preprocessor and model
            preprocessor_obj_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")
            model_path = os.path.join("artifacts/model_trainer", "model.pkl")

            # Load preprocessor and model
            processor = load_obj(preprocessor_obj_path)
            model = load_obj(model_path)

            # Transform features and make predictions
            scaled = processor.transform(features)
            pred = model.predict(scaled)

            # Debugging output
            logger.info(f"Raw Prediction Result: {pred}")

            # Handle prediction result
            if isinstance(pred, (np.ndarray, list)):
                pred = np.array(pred).flatten()  # Flatten if it's a 2D array or list
            if len(pred) == 1:
                pred = pred[0]  # Extract single value if prediction is an array with one element

            # Convert result to appropriate format (e.g., string or categorical)
            if isinstance(pred, float):
                # Assuming a binary classification, map the result to 'hate' or 'normal'
                result = 'good' if pred > 0.5 else 'bad'
            else:
                result = pred

            logger.info(f"Processed Prediction Result: {result}")

            return result

        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise CustomException(e, sys)
