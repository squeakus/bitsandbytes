import math
  
def dot(v,w):
    x,y,z = v
    X,Y,Z = w
    return x*X + y*Y + z*Z
  
def length(v):
    x,y,z = v
    return math.sqrt(x*x + y*y + z*z)
  
def vector(b,e):
    x,y,z = b
    X,Y,Z = e
    return (X-x, Y-y, Z-z)
  
def unit(v):
    x,y,z = v
    mag = length(v)
    return (x/mag, y/mag, z/mag)
  
def distance(p0,p1):
    return length(vector(p0,p1))
  
def scale(v,sc):
    x,y,z = v
    return (x * sc, y * sc, z * sc)
  
def add(v,w):
    x,y,z = v
    X,Y,Z = w
    return (x+X, y+Y, z+Z)
