from math import pi, radians, cos, sin

point = [0, 10, 0]
x = [0, 10, 0]

for i in [0, 90, 180, 270, 360]:
    print i
    ang = radians(i)
    rotated_x = (point[0] * cos(ang)) + (point[1] * sin(ang))
    rotated_y = (- point[0] * sin(ang)) + (point[1] * cos(ang))
    print "x:", rotated_x, "y:", rotated_y
    new_x = ((x[0]) * cos(ang) +(x[1])*sin(ang))
    new_y = ((-x[0]) * sin(ang) +(x[1])*cos(ang))

print "2pi:", str((2 * pi))
    

