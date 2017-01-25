from pyproj import Proj, transform

inProj = Proj(init='epsg:29902')
#outProj = Proj(init='epsg:4326')
outProj = Proj(init='epsg:3587')
x1,y1 = 315904.00, 234671.00
x2,y2 = transform(inProj,outProj,x1,y1)
print x2,y2
