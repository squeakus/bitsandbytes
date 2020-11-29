def main():
    w = 100
    h = 50
    x = 20
    y = 10

    print("original:", x, y, w, h)
    for i in range(4):
        x, y, w, h = rotate_counterclockwise(x, y, w, h)
        print(x, y, w, h)


def rotate_clockwise(x, y, w, h):
    xnew = (h - 1) - y
    ynew = x
    return xnew, ynew, h, w


def rotate_counterclockwise(x, y, w, h):
    xnew = y
    ynew = (w - 1) - x
    return xnew, ynew, h, w


if __name__ == "__main__":
    main()