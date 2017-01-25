#!/usr/bin/python                                                               
import sys

expression = ''
for line in sys.stdin:
    line = line.rstrip("\n")
    expression += line
expression = list(expression)    

result =0
stack =[]
print "expression:",expression
for token in expression:
    print stack
    print "stack length!",len(stack)
    print "token",token
    if(token == '+'):
        result = int(stack.pop()) + int(stack.pop())
        stack.append(result)
    elif(token == '-'):
        var1 = int(stack.pop())
        var2 = int(stack.pop())
        result =  var2 - var1
        stack.append(result)
    elif(token == '*'):
        result = int(stack.pop()) * int(stack.pop())
        stack.append(result)
    else:
        stack.append(token)
print result

