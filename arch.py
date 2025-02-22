import random
from engine import Value 

class Module:

    def zero_grad(self):
        
        for parameter in self.parameters:
            parameter.grad = 0.0
        
    def parameters(self):

        return []

class Conv2d(Module):

    def __init__(self, inch, outch, kernel, stride):

        self.inch = inch
        self.outch = outch
        self.kernel = kernel
        self.stride = stride
        self.w = [Value(random.uniform(-1,1)) for _ in range(kernel * kernel * outch * inch)]
        self.b = [Value(random.uniform(-1,1)) for _ in range(outch)]


    def __call__(self, x):
        
        assert self.outch == len(x[:]), "The number of in-channels must match the outer dimension of the given array"
        
        # The input 'x' is of the format batch size, in-channels, hieght, width
        hi = len(x[0][:])
        wi = len(x[0][0][:])
        hf = (hi - self.kernel) // self.stride + 1
        wf = (wi - self.kernel) // self.stride + 1
        
        # To hold the output of the convolution operation initialized with 0s 
        out = [[[Value(0) for _ in range(wf)] for _ in range(hf)] for _ in range(self.outch)]
        
        # 'oc' corresponds to a specific out-channel.
        # 'ic' corresponds to a specific in-channel.
        for oc in range(self.outch):
            for i in range(0, hi - self.kernel + 1, self.stride):
                for j in range(0, wi - self.kernel + 1, self.stride):

                    # Step amounts equal to stride over the input.
                    curr = self.b[oc]
                    for ic in range(self.inch):

                        # For every 'ic' in 'in-channels', apply the kernel tranformation. 
                        # For every element in the self.kernel * self.kernel sized kernel, multiply accordingly. 
                        for ki in range(self.kernel):
                            for kj in range(self.kernel):

                                # Get the exact 'weight' corresponding to the current element. 
                                weightid = oc * self.inch * self.kernel * self.kernel + ic * self.kernel * self.kernel + ki * self.kernel + kj
                                
                                # Current value to multiply from the input matrix is at the position:
                                # current batch, current in-channel (ic), amount moved in the current row (i + ki), amount moved in the current column (j + kj)
                                curr += self.w[weightid] * x[ic][i + ki][j + kj]
                    
                    # Update the value in the output matrix
                    out[oc][i // self.stride][j // self.stride] = curr
        
        return out       
     
    def parameters(self):
        return self.w + self.b


class Linear(Module):
    
    """
        Similar to the torch.nn.Linear class. 

        Takes as input the number of in-channels and out-channels
    """
    
    def __init__(self, inch, outch):

        self.inch = inch
        self.outch = outch
        self.w = [Value(random.uniform(-1,1)) for _ in range(inch * outch)]
        self.b = [Value(random.uniform(-1,1)) for _ in range(outch)]

    def __call__(self, x):

        x = [i if isinstance(i, Value) else Value(i) for i in x]  
        assert len(x) == self.inch, "The two must be of the same length"

        out = []
        for i in range(self.outch):
            
            element = self.b[i]

            for j in range(self.inch):
                element += self.w[i * self.inch + j] * x[j]
            
            out.append(element)
        
        return out
        
    
    def parameters(self):
        return self.w + self.b

class Neuron(Module):
    
    def __init__(self, n_connections):
         
        self.w = [Value(random.uniform(-1,1) for _ in range(n_connections))]
        self.b = Value(random.uniform(-1,1))

    def __call__(self, x):
        
        # pass the value forward: w * x + b
        activation = sum((i*j for i,j in zip(self.w, x)), self.b)
        out = activation.tanh()
        return out 

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, n_connections, n_neurones):
        self.neurons = [Neuron(n_connections) for _ in range(n_neurones)]

    def __call__(self, x):
        
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    
    def parameters(self):
        return [neuron.parameters() for neuron in self.neurons]