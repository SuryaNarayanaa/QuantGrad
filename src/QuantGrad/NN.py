import random
from QGrad import Value, draw_dot
class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1,1), _label= 'w') for _ in range(nin)]
        self.bias    = Value(random.uniform(-1,1), _label= 'b')
    
    def __call__(self, x):
        activation = sum([wi*xi for wi,xi in zip(self.weights, x)])+self.bias
        out =activation.tanh()
        return out
    
    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self,nin,nout):
        self.neurons= [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) ==1 else outs
        
    def parameters(self):
        return [n for neuron in self.neurons for n in neuron.parameters() ]
        
class MLP:
    def __init__(self,nin,nouts):
        sz= [nin]+nouts #prepending the nin to nouts list
        self.layers= [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]


    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [neurons for layer in self.layers for neurons in layer.parameters() ]
    
    def step(self ,step):
        for p in self.parameters():
            p.data += -step*p.grad
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def MSE(self, Y, predictions):
        loss = sum([(ypred-ygt)**2 for ypred,ygt in zip(predictions, Y)])
        return loss

        