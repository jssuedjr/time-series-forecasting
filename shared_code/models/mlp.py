import numpy as np
from .model_interface import ForecastModel

'''
if training the model using returns/log-returns, make sure to transform 
the prediction back to the raw closing price inside the MLP model


The scaling parameters must be estimated from the training window only.
'''

# multi-layer perceptron
class MLPModel(ForecastModel):

    def __init__(
        self,
        n_lags=20,
        hidden_size=16,
        learning_rate=0.001,
        epochs=100,
        random_state=42
    ):
        ...

    def fit(self, train):
        # 1. Convert prices into log returns
        # 2. Construct lagged training examples
        # 3. Fit scaling parameters on training data
        # 4. Initialize neural network weights
        # 5. Train using gradient descent
        ...

    def predict(self, horizon):
        # 1. Retrieve the latest lagged returns
        # 2. Predict future log returns
        # 3. Convert predictions back into prices
        # 4. Return an array of shape (horizon,)
        ...