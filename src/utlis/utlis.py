# from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
import os, sys
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# save file into folders
def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb')as file:
            pickle.dump(obj,file)

    except Exception as e:
        raise CustomException(e,sys)
    

def model_evaluate(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        for model_name, model in models.items():
            param = params[model_name]

            gs = GridSearchCV(model, param, cv=5)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # make prediction
            y_pred = model.predict(X_test)
            test_model_accuracy = accuracy_score(y_test, y_pred)

            # Store the accuracy in the report dictionary
            report[model_name] = test_model_accuracy

        return report
    except Exception as e:
        raise CustomException(e, sys)

    

def load_obj(file_path):
    try:
        with open(file_path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e,sys)