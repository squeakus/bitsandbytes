x = 0.00000001

for itr in range(999999999):
    if itr % 1000000 == 0:
	print "count:", itr
    x += 0.00000001
    a = x * x
    b = x ** 2
    diff = a - b
    if  diff != 0:
        print "error", a, b, diff

