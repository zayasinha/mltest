import sys
sys.path.append(r'c:\apple')
from src.mltest.pipelines.training_pipeline import TrainingPipeline
print('Re-running full training pipeline with RandomizedSearchCV...')
pipeline = TrainingPipeline()
model_path, preprocessor_path = pipeline.run_pipeline()
print('Pipeline finished. Model saved at:', model_path)
print('Preprocessor saved at:', preprocessor_path)
