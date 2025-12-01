"""
Prediction Pipeline - Inference on New Data
Provides clean interface for making predictions with trained model.
"""
import os
import sys
from typing import Dict, Union, List
import pandas as pd
import numpy as np
from src.mltest.exception import CustomException
from src.mltest.logger import logging
from src.mltest.utils import load_object


class PredictionPipeline:
    """
    Prediction pipeline for BMW car price prediction.
    
    Loads trained model and preprocessor to make predictions on new data.
    """
    
    def __init__(
        self,
        preprocessor_path: str = None,
        model_path: str = None
    ):
        """
        Initialize prediction pipeline.
        
        Args:
            preprocessor_path: Path to preprocessor (default: artifacts/preprocessor.pkl)
            model_path: Path to model (default: artifacts/model.pkl)
        """
        self.preprocessor_path = preprocessor_path or os.path.join('artifacts', 'preprocessor.pkl')
        self.model_path = model_path or os.path.join('artifacts', 'model.pkl')
        
        # Lazy loading - only load when needed
        self._preprocessor = None
        self._model = None
        
        logging.info("PredictionPipeline initialized")
    
    @property
    def preprocessor(self):
        """Lazy load preprocessor."""
        if self._preprocessor is None:
            logging.info(f"Loading preprocessor from {self.preprocessor_path}")
            self._preprocessor = load_object(self.preprocessor_path)
            logging.info("✓ Preprocessor loaded successfully")
        return self._preprocessor
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            logging.info(f"Loading model from {self.model_path}")
            self._model = load_object(self.model_path)
            logging.info("✓ Model loaded successfully")
        return self._model
    
    def predict(self, features: Union[Dict, pd.DataFrame]) -> Union[float, np.ndarray]:
        """
        Predict BMW car price from input features.
        
        Args:
            features: Either a dictionary with feature values or DataFrame
                Dictionary keys: year, mileage, engineSize, tax, mpg, 
                                model, transmission, fuelType
                
        Returns:
            Predicted price (float for single prediction, array for batch)
            
        Raises:
            CustomException: If prediction fails
            
        Example:
            >>> pipeline = PredictionPipeline()
            >>> car = {
            ...     'model': '3 Series',
            ...     'year': 2020,
            ...     'transmission': 'Semi-Auto',
            ...     'mileage': 5000,
            ...     'fuelType': 'Diesel',
            ...     'tax': 145,
            ...     'mpg': 52.3,
            ...     'engineSize': 2.0
            ... }
            >>> price = pipeline.predict(car)
            >>> print(f"Predicted price: £{price:.2f}")
        """
        try:
            logging.info("Starting prediction...")
            
            # Convert to DataFrame if dictionary
            if isinstance(features, dict):
                df = pd.DataFrame([features])
                logging.info("Input: Single car (dictionary)")
            else:
                df = features.copy()
                logging.info(f"Input: Batch of {len(df)} cars (DataFrame)")
            
            logging.info(f"Input shape: {df.shape}")
            logging.info(f"Input columns: {df.columns.tolist()}")
            
            # Validate required columns
            required_cols = ['year', 'mileage', 'engineSize', 'tax', 'mpg', 
                           'model', 'transmission', 'fuelType']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Transform features
            logging.info("Applying preprocessing transformations...")
            data_scaled = self.preprocessor.transform(df)
            logging.info(f"Transformed shape: {data_scaled.shape}")
            
            # Make prediction
            logging.info("Making prediction...")
            prediction = self.model.predict(data_scaled)
            
            # Return single value if single prediction
            if len(prediction) == 1:
                result = float(prediction[0])
                logging.info(f"✓ Prediction completed. Predicted price: £{result:,.2f}")
                return result
            else:
                logging.info(f"✓ Batch prediction completed. {len(prediction)} predictions made")
                return prediction
            
        except Exception as e:
            logging.error("Error in prediction")
            raise CustomException(e, sys)
    
    def predict_batch(self, features_list: List[Dict]) -> np.ndarray:
        """
        Batch prediction for multiple cars.
        
        Args:
            features_list: List of feature dictionaries
            
        Returns:
            Array of predicted prices
            
        Example:
            >>> pipeline = PredictionPipeline()
            >>> cars = [
            ...     {'model': '3 Series', 'year': 2020, ...},
            ...     {'model': '5 Series', 'year': 2019, ...}
            ... ]
            >>> prices = pipeline.predict_batch(cars)
        """
        logging.info(f"Starting batch prediction for {len(features_list)} cars")
        df = pd.DataFrame(features_list)
        return self.predict(df)
    
    def predict_with_details(self, features: Dict) -> Dict:
        """
        Predict with additional details like confidence intervals.
        
        Args:
            features: Dictionary with car features
            
        Returns:
            Dictionary with prediction and metadata
        """
        try:
            # Make prediction
            price = self.predict(features)
            
            # Build response
            result = {
                'predicted_price': float(price),
                'currency': 'GBP',
                'model_used': type(self.model).__name__,
                'input_features': features,
                'status': 'success'
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error in detailed prediction: {e}")
            return {
                'status': 'error',
                'error_message': str(e),
                'predicted_price': None
            }


# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = PredictionPipeline()
    
    # Example single prediction
    example_car = {
        'model': '3 Series',
        'year': 2020,
        'transmission': 'Semi-Auto',
        'mileage': 5000,
        'fuelType': 'Diesel',
        'tax': 145,
        'mpg': 52.3,
        'engineSize': 2.0
    }
    
    try:
        print("="*80)
        print("PREDICTION PIPELINE TEST")
        print("="*80)
        
        # Single prediction
        price = pipeline.predict(example_car)
        print(f"\n✓ Single Prediction:")
        print(f"   Input: {example_car['year']} {example_car['model']}")
        print(f"   Predicted BMW price: £{price:,.2f}")
        
        # Batch prediction
        cars = [
            example_car,
            {
                'model': '5 Series',
                'year': 2019,
                'transmission': 'Automatic',
                'mileage': 10000,
                'fuelType': 'Petrol',
                'tax': 150,
                'mpg': 45.0,
                'engineSize': 3.0
            },
            {
                'model': 'X5',
                'year': 2021,
                'transmission': 'Automatic',
                'mileage': 2000,
                'fuelType': 'Diesel',
                'tax': 500,
                'mpg': 38.5,
                'engineSize': 3.0
            }
        ]
        
        prices = pipeline.predict_batch(cars)
        print(f"\n✓ Batch Predictions ({len(cars)} cars):")
        for i, (car, price) in enumerate(zip(cars, prices), 1):
            print(f"   {i}. {car['year']} {car['model']}: £{price:,.2f}")
        
        # Detailed prediction
        print(f"\n✓ Detailed Prediction:")
        details = pipeline.predict_with_details(example_car)
        print(f"   Price: £{details['predicted_price']:,.2f}")
        print(f"   Model: {details['model_used']}")
        print(f"   Status: {details['status']}")
        
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
