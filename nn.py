import numpy as np
from nnfs.datasets import spiral_data
class DenseLayer:
    
    def __init__(self,n_inputs,n_neuros) :
        # Intialize the weights and the biases
        self.weights = np.random.randn(n_inputs,n_neuros)*0.01
        self.biases = np.zeros((1,n_neuros))
        
    def forward(self,inputs):
        # Forward pass
        self.output = np.dot(inputs,self.weights) + self.biases


class ReLU:
    
   def forward(self,inputs):
       # Forward pass with activation function
       self.output = np.maximum(0,inputs)
       
class SoftMax:
    
    def forward(self,inputs):
        inputs = inputs - np.max(inputs,axis=1,keepdims=True)
        self.output = np.exp(inputs)/np.sum(np.exp(inputs),axis=1,keepdims=True)
        
# Create dataset

X,y = spiral_data(samples=100,classes=3)

dense1 = DenseLayer(2,3)

dense1.forward(X)

softmax_activation = SoftMax()

softmax_activation.forward(dense1.output)
print(softmax_activation.output)