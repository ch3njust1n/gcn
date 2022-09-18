class GradientDescent(object):
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate
        
        
    def zero_gradients(self):
        for layer in self.parameters:
            layer.zero_gradients()
    
    
    def step(self):
        for layer in self.parameters:
            layer.weights -= self.learning_rate * layer.gradients