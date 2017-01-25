"""A simple odometry module for a two wheeled robot."""
import math


def main():
    """
    Specify the input turns and record the change in location. Each move
    is considered an arc. Errors will accumulate during the run.
    """
    # the heading is in radians! 1 rad = 57.2 deg
    l1 = 0
    r1 = 4.7  # right angle
    l2 = 100
    r2 = 100

    x, y, h = 0, 0, 0
    x, y, h = compute_location(l1, r1, x, y, h)
    x, y, h = compute_location(l2, r2, x, y, h)


def compute_location(left_encoder, right_encoder, xcoord, ycoord, heading):
    """Calculate the new position and heading given a number of wheel turns."""
    counts_per_cm = 1
    wheelbase = 3

    distance = (left_encoder + right_encoder)/2
    heading += (right_encoder - left_encoder)/wheelbase

    distance = distance / counts_per_cm
    xcoord = distance * math.sin(heading)
    ycoord = distance * math.cos(heading)
    print "distance:", distance, "heading:", heading
    print "x:", xcoord, "y:", ycoord
    return xcoord, ycoord, heading

if __name__ == '__main__':
    main()
