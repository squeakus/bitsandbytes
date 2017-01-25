import math
  
def dot(v,w):
    x,y = v
    X,Y = w
    return x*X + y*Y
  
def length(v):
    x,y = v
    return math.sqrt(x*x + y*y)
  
def vector(b,e):
    x,y = b
    X,Y = e
    return (X-x, Y-y)
  
def unit(v):
    x,y = v
    mag = length(v)
    return (x/mag, y/mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y = v
    return (x * sc, y * sc)
  
def add(v,w):
    x,y = v
    X,Y = w
    return (x+X, y+Y)
