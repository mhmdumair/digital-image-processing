import numpy as np

a = [1,2,3,4,5,6,7,8,9]
a = np.array(a)

b = np.array([9,8,7,6,5,4,3,2,1], dtype=complex)
zero = np.zeros(10)
line = np.linspace(10,18,5)
space = np.arange(10,20,2)

# print("zeros",zero)
# print("Line",line)
# print("space",space)

# Rando values
r1 = np.random.normal(0,1,10)  # mean , sd , size
# print(r1)
# print(np.min(r1))
# print(np.max(r1))
# print(np.mean(r1))
# print(np.median(r1))
# print(np.std(r1))

# Matrix

a = np.array([[1,2],[3,4]])
b = np.array([[5,6],[7,8]])
c=  a*b
d = a.dot(b)
e = np.matmul(a,b)
print(c)
print(e)
print(d)

# print(a[0][0])
# print(a[:,:])
