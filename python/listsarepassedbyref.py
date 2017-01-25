def multiply(val):
	val[0] = val[0] * 2
	return val

x = [10,10]
print "before ", x

y = multiply(x)
print "after  ", x
print y
