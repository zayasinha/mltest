import sys

from src.mltest.logger import logging
from src.mltest.exception import CustomException
from src.mltest.components.data_ingestion import DataIngestion
from src.mltest.components.data_transformation import (
    DataTransformationConfig,
    DataTransformation,
)
from src.mltest.components.model_trainer import (
    ModelTrainerConfig,
    ModelTrainer,
)


if __name__ == "__main__":
    logging.info("Pipeline execution started")

    try:
        # 1. Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # 2. Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path,
            test_data_path,
        )

        # 3. Model Training
        model_trainer = ModelTrainer()
        best_model_path, model_report = model_trainer.initiate_model_trainer(
            train_arr,
            test_arr,
        )

        logging.info("Training completed successfully")
        logging.info("Best model saved at: %s", best_model_path)
        print("Best model path:", best_model_path)
        # Optionally inspect model_report here

    except Exception as e:
        logging.info("Custom Exception raised in main pipeline")
        raise CustomException(e, sys)
