from src.mltest.logger import logging
from src.mltest.exception import CustomException
from src.mltest.utils import read_sql_data
from src.mltest.components.data_ingestion import DataIngestion
from src.mltest.components.data_ingestion import DataIngestionConfig
import sys

if __name__ == "__main__":
    logging.info("Starting the MLTest application...")
    # Application logic would go here
    logging.info("MLTest application finished.")

try:
   # data_ingestion_config = DataIngestionConfig()
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()
    
except Exception as e:
    logging.info("An exception occurred.")
    raise CustomException(e, sys)       