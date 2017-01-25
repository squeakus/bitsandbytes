import inspect

class my_class():
    classvar = 10
    def __init__(self):
	self.myvar = 5
    def moo():
	print "moo"
    def test_moo():
	return True

print inspect.getmembers(my_class)
