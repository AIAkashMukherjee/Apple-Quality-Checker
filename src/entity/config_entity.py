from dataclasses import dataclass
import os


@dataclass
class DataIngestionConfig:
    train_file_path=os.path.join('artifacts/data_ingestion','train.csv')
    test_file_path=os.path.join('artifacts/data_ingestion','test.csv')
    raw_file_path=os.path.join('artifacts/data_ingestion','raw.csv')

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/data_transformation', 'preprocessor.pkl')


@dataclass 
class ModelTrainerConfig:
    train_model_file_path=os.path.join('artifacts/model_trainer','model.pkl')    