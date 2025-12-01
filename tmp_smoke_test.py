import numpy as np
from mltest.components.model_trainer import ModelTrainer

# create synthetic regression data (100 samples, 5 features)
X = np.random.rand(100, 5)
y = X @ np.array([1.2, -0.5, 0.3, 0.0, 0.6]) + np.random.randn(100) * 0.1
data = np.hstack([X, y.reshape(-1,1)])
# simple train/test split
train, test = data[:80], data[80:]
trainer = ModelTrainer()
print('Starting training...')
model_path = trainer.initiate_model_trainer(train, test)
print('Saved model to:', model_path)
