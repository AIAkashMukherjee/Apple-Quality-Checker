import sys
import pandas as pd
import numpy as np
from src.logger.custom_logging import logger
from src.exceptions.expection import CustomException
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from src.entity.config_entity import DataTransformationConfig
from sklearn.compose import ColumnTransformer
from src.utlis.utlis import save_obj


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):
        try:
            logger.info('Data Transformation started')

            numerical_features = ['Size','Weight',	'Sweetness',	'Crunchiness',	'Juiciness',	'Ripeness'	,'Acidity']

            num_pipeline=Pipeline(
                steps = [
                ("imputer", SimpleImputer(strategy = 'median')),
                ("scaler", StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
                ('Num_pipeline', num_pipeline,numerical_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)    
        

    def initate_data_transformation(self,train_path,test_path):
        try:
            train_data=pd.read_csv(train_path)
            test_data=pd.read_csv(test_path)

            train_data.dropna(inplace=True)
            test_data.dropna(inplace=True)

            preprocessor_obj=self.get_data_transformation()

            target_columns='Quality'
            drop_columns=[target_columns,'A_id']

            logger.info("Splitting train data into dependent and independent features")
            input_feature_train_data = train_data.drop(drop_columns, axis = 1)
            traget_feature_train_data = train_data[target_columns]

            logger.info("Splitting test data into dependent and independent features")
            input_feature_test_data = test_data.drop(drop_columns, axis = 1)
            traget_feature_test_data = test_data[target_columns]

            # Apply preprocessor object on our train data and test data
            input_train_arr=preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor_obj.transform(input_feature_test_data)

             # Apply preprocessor object on our train data and test data
            train_array=np.c_[input_train_arr,np.array(traget_feature_train_data)]

            test_array=np.c_[input_test_arr,np.array(traget_feature_test_data)]

            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessor_obj)

            return (train_array,
                    test_array,
                    self.data_transformation_config.preprocessor_obj_file_path)



        except Exception as e:
            raise CustomException(e,sys)    
