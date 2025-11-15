from src.mltest.logger import logging
from src.mltest.exception import CustomException
#from src.mltest.utils import read_sql_data
from src.mltest.components.data_ingestion import DataIngestion
from src.mltest.components.data_ingestion import DataIngestionConfig
from src.mltest.components.data_transformation import DataTransformationConfig, DataTransformation
import sys

if __name__ == "__main__":
    try:
        logging.info("Starting the MLTest application...")
        
        # data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        # data_transformation_config = DataTransformationConfig()
        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        logging.info("MLTest application finished.")
        
    except Exception as e:
        logging.info("An exception occurred.")
        raise CustomException(e, sys)       