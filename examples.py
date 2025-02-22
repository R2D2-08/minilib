import random
from engine import Value
from arch import Module, Linear, Conv2d

a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
d = Value(5.0, label ='d')
e = a * b
e.label = 'e'
f = c / d
f.label = 'f'
g = c * d + e
g.label = 'g'
h = f - g; h.label = 'h'

h.backward()
print(f"{a}\n{b}\n{c}\n{d}\n{e}\n{f}\n{g}\n{h}")

example_array1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
example_array2 = [[[13,31,32],[34,33,35]],[[93,10,11],[23,73,43]],[[65,86,95],[83,53,52]],[[63,89,15],[49,59,25]]]

z = Linear(10, 20)
print(z(example_array1))

y = Conv2d(3, 4, 2, 1)
print(y(example_array2))