import math

class Value:

	def __init__(self, data, _children = (), _op = '', label = ''):
		
		self.data = data
		self.grad = 0.0
		
		# For every node, 'backward' to propagate deeper into the graph and set the graidents
		self._backward = lambda: None

		self._prev = set(_children)
		self._op = _op
		self.label = label

	def __repr__(self):

		return f"Value(data = {self.data}, gradient = {self.grad})"

	def __add__(self, other):
		
		# Ensure the other is a Value object:
		other = other if isinstance(other, Value) else Value(other)

		out = Value(self.data + other.data, (self, other), '+')

		def _backward():
		
			'''
			out = a + b

			d(loss)/d(a) = d(loss)/d(out) * d(out)/d(a)
			d(loss)/d(b) = d(loss)/d(out) * d(out)/d(b)

			d(out)/d(a) = 1.0
			d(out)/d(b) = 1.0

			'''
		
			self.grad += 1.0 * out.grad
			other.grad += 1.0 * out.grad

		# Set and Make the call to the backward pass from the current node
		out._backward = _backward	
		
		return out
		
	def __mul__(self, other):

		# Ensure the other is a Value object:
		other = other if isinstance(other, Value) else Value(other)
		
		out = Value(self.data * other.data, (self, other), '*')

		def _backward():
		
			'''
			out = a * b

			d(loss)/d(a) = d(loss)/d(out) * d(out)/d(a)
			d(loss)/d(b) = d(loss)/d(out) * d(out)/d(b)

			d(out)/d(a) = b
			d(out)/d(b) = a

			'''
		
			self.grad += other.data * out.grad
			other.grad += self.data * out.grad

		# Set and Make the call to the backward pass from the current node
		out._backward = _backward	
		
		return out

	def __pow__(self, other):

		# 'other' here is an integer or a float
		assert isinstance(other, (int,float))

		out = Value(self.data ** other, (self,), 'pow')	

		def _backward():

			'''
			out = a ** b

			d(loss)/d(a) = d(loss)/d(out) * d(out)/d(a)			
			d(out)/d(a) = b * a ** (b-1)
			
			'''
			self.grad += out.grad * other * self.data ** (other - 1)

		out._backward = _backward

		return out
	
	def relu(self):

		# Ensure the other is a Value object:
		other = other if isinstance(other, Value) else Value(other)
		
		out = Value(max(self.data, 0), (self, ), 'ReLU')

		def _backward():
			
			'''
			out = max(a, 0)

			d(loss)/d(a) = 0 if a < 0
			d(loss)/d(a) = d(loss)/d(out) * 1 if a > 0

			'''

			self.grad += out.grad * 1.0 if self.data > 0 else 0

		out._backward = _backward

		return out

	def tanh(self):

		# Ensure the other is a Value object:
		other = other if isinstance(other, Value) else Value(other)
		
		x = self.data
		t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
		out = Value(t, (self, ), 'tanh')

		def _backward():
			
			'''
			out = tanh(x)

			d(loss)/d(x) = d(loss)/d(out) * d(tanh(x))/d(x)

			d(tanh(x))/d(x) = (1 - tanh(x)**2)
		
			'''
	
			self.grad = (1 - t**2) * out.grad

		out._backward = _backward

		return out

	def backward(self):
		
		dag = []
		visited = set()

		def fill_dag(node):
	
			# Explore the directed acyclic graph from the rightmost node for every other node to perform backpropagation on.
			
			if node not in visited: 
			
				visited.add(node)
			
				for child in node._prev:
					fill_dag(child)
			
				dag.append(node)

		fill_dag(self)

		dag = reversed(dag)

		# Set the gradient of the rightmost node to 1.0
		# d(loss)/d(loss) = 1.0

		self.grad = 1.0
		
		for node in dag:		
			node._backward()

	def __neg__(self):
		return self * -1
	
	def __sub__(self, other):
		return self + (-other)
	
	def __rsub__(self, other):
		return other + (-self)
	
	def __radd__(self, other):
		return other + self
	
	def __rmul__(self, other):
		return self * other
	
	def __truediv__(self, other):
		return self * (other ** -1)
	
	def __rtruediv__(self, other):
		return other * (self ** -1)