size1 = 500
size2 = 500
n=64
box=((-2,1.25),(0.5,-1.25))
plus = size[1]+size[0]
uleft = box[0]
lright = box[1]
xwidth = lright[0] - uleft[0]
ywidth = uleft[1] - lright[1]

for y in xrange(size[1]):
        coords = (uleft[0] + (x/size[0]) * (xwidth),uleft[1] - (y/size[1]) * (ywidth))
        z = complex(coords[0],coords[1])
        o = complex(0,0)
        dotcolor = 0  # default, convergent
        for trials in xrange(n):
            if abs(o) <= 2.0:
                o = o**2 + z
            else:
                dotcolor = trials
                break  # diverged
        im.putpixel((x,y),dotcolor)
