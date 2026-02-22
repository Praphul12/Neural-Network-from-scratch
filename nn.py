import numpy as np
from nnfs.datasets import spiral_data
class DenseLayer:
    
    def __init__(self,n_inputs,n_neuros) :
        
        self.weights = np.random.randn(n_inputs,n_neuros)*0.01
        self.biases = np.zeros((1,n_neuros))
        
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases



# Create dataset

X,y = spiral_data(samples=100,classes=3)

dense1 = DenseLayer(2,4)

dense1.forward(X)

print(dense1.output)