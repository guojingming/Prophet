import numpy

base = 20000 * 0.8

rate = 1.01

a = numpy.power(1.01, 6 * 12)

year = 7

result = base * numpy.power(a, year) * 6.5

print(result)