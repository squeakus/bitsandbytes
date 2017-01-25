import math, sys

height = 100.0
horizfov = 81.0 
vertfov = 66.0

#height = adjacent, base = opposite, angled = hypoteneuse
#find base: tangent(1/2FOV) = opposite / adjacent
# next step, calc the trapezoid
def main(height):
    width = calc_base(height, horizfov)
    length = calc_base(height, vertfov)
    print "width %.02f length %.02f" % (width, length)
    area = calc_area(length, width)
    print "area", area
    extrema = calc_extrema(length, width)
    print "furthest point", extrema
    print "max overlap", extrema * 2
def rightangledbase(height, fov):
    angle = fov / 2
    base = math.tan(math.radians(angle)) * height
    return base

def calc_base(height, fov):
    halfbase = rightangledbase(height, fov)
    base = round(halfbase * 2,2)
    return base

def calc_area(length, width):
    area = round(length * width, 2)
    return area

def calc_extrema(length, width):
    """computing furthest possible distance from the midpoint"""
    length = length / 2
    width = width / 2
    hypo = round(math.sqrt(width**2 + length**2),2)
    return hypo
    


if __name__=='__main__':

    if len(sys.argv) < 2:
        print ("Usage %s <height> " % sys.argv[0])
        sys.exit(1)
    main(float(sys.argv[1]))
