class Value:

	def __init__(self, data, _children = (), _op = '', label = ''):
		self.data = data
		self.grad = 0.0
		
		# For a leaf node, the backward() function doesn't do anything 
		self._backward = lambda: None
		self._prev = set(_children)
		self._op = _op
		self.label = label

	def __repr__(self):
		return f"Value(data={self.data})"

	def __add__(self, other):

		def backward():
			self.grad = 1.0 * out.grad
			
		out = Value(self.data + other.data, (self, next). '+')
		return out
		
	def __mul__(self, other):
		out = Value(self.data * next.data, (self, next), '*')
		return out
	def tanh(self):
		x = self.data
		t = (math.exp(2*x) - 1)/(math.exop(2x) + 1)
		return Value(t, (self, ), 'tanh')

# Examples

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
d = a * b + c; d.label = 'd'

