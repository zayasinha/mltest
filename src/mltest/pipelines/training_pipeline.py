"""
Training Pipeline - Orchestrates Complete ML Workflow
Coordinates data ingestion, transformation, and model training.
"""
import os
import sys
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from src.mltest.exception import CustomException
from src.mltest.logger import logging
from src.mltest.components.data_ingestion import DataIngestion
from src.mltest.components.data_transformation import DataTransformation
from src.mltest.components.model_trainer import ModelTrainer


@dataclass
class TrainingPipelineConfig:
    """Configuration for training pipeline."""
    pipeline_name: str = "MLTest Training Pipeline"
    artifacts_dir: str = "artifacts"


class TrainingPipeline:
    """
    Complete training pipeline orchestrator.
    
    Executes:
    1. Data Ingestion
    2. Data Transformation
    3. Model Training
    """
    
    def __init__(self, config: TrainingPipelineConfig = None):
        """
        Initialize training pipeline.
        
        Args:
            config: TrainingPipelineConfig (uses default if None)
        """
        self.config = config or TrainingPipelineConfig()
        
        # Initialize components
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()
        
        logging.info(f"Initialized {self.config.pipeline_name}")
    
    def run_pipeline(self) -> Tuple[str, str]:
        """
        Execute complete training pipeline.
        
        Returns:
            Tuple of (model_path, preprocessor_path)
            
        Raises:
            CustomException: If any pipeline stage fails
        """
        logging.info("="*80)
        logging.info(f"STARTING {self.config.pipeline_name.upper()}")
        logging.info("="*80)
        
        try:
            # Stage 1: Data Ingestion
            logging.info("\n" + "="*80)
            logging.info("STAGE 1: DATA INGESTION")
            logging.info("="*80)
            
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            
            logging.info(f"✓ Train data: {train_data_path}")
            logging.info(f"✓ Test data: {test_data_path}")
            
            # Stage 2: Data Transformation
            logging.info("\n" + "="*80)
            logging.info("STAGE 2: DATA TRANSFORMATION")
            logging.info("="*80)
            
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )
            
            logging.info(f"✓ Preprocessor: {preprocessor_path}")
            logging.info(f"✓ Train array shape: {train_arr.shape}")
            logging.info(f"✓ Test array shape: {test_arr.shape}")
            
            # Stage 3: Model Training
            logging.info("\n" + "="*80)
            logging.info("STAGE 3: MODEL TRAINING")
            logging.info("="*80)
            
            model_path, model_report = self.model_trainer.initiate_model_trainer(
                train_arr, test_arr
            )
            
            logging.info(f"✓ Model: {model_path}")
            
            # Pipeline completion
            logging.info("\n" + "="*80)
            logging.info(f"{self.config.pipeline_name.upper()} COMPLETED SUCCESSFULLY ✓")
            logging.info("="*80)
            logging.info(f"Model saved at: {model_path}")
            logging.info(f"Preprocessor saved at: {preprocessor_path}")
            logging.info(f"Artifacts directory: {self.config.artifacts_dir}")
            logging.info("="*80)
            
            return model_path, preprocessor_path
            
        except Exception as e:
            logging.error(f"Error in {self.config.pipeline_name}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Run training pipeline
    try:
        pipeline = TrainingPipeline()
        model_path, preprocessor_path = pipeline.run_pipeline()
        
        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETED")
        print("="*80)
        print(f"Model: {model_path}")
        print(f"Preprocessor: {preprocessor_path}")
        print("="*80)
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)
