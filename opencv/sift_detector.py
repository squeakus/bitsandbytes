"""
A sift detection example
SIFT's patent has expired in last July. in versions > 4.4
So remember to upgrade:
pip install opencv-python --upgrade
"""

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


def main():
    main_image = "1.42_1.jpg"
    target_image = "target1.png"
    draw_keypoints(main_image)
    match_coords(main_image, target_image)


def match_coords(main_image, target_image):
    MIN_MATCH_COUNT = 10

    img1 = cv2.imread(target_image, 0)
    img2 = cv2.imread(main_image, 0)
    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    print("Matches:", len(good))
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        points = [np.int32(dst)]
        center_point = np.average(points, axis=1).astype(int)[0][0]
        print(center_point)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        img2 = cv2.circle(img2, tuple(center_point), 10, 255, 2)

        plt.imshow(img2, "gray"), plt.show()
        rows, cols = img2.shape
        bestfit = 0
        bestave = 255

        for y in range(rows):
            rowvals = img2[y]
            average = int(ave(rowvals))
            std_dev = int(std(rowvals, average))
            print("ave", average, "+-", std_dev)
            if (std_dev < 10) and (average < 150):
                bestfit = y
                bestave = average
                beststd = std_dev
                break

        img2 = cv2.line(img2, (0, bestfit), (cols, bestfit), 255, 3)
        # plt.imsave('line{0:03d}.png'.format(y), img2)
        img2 = cv2.line(img2, (center_point[0], center_point[1]), (center_point[0], bestfit), 255, 3)
        plt.imshow(img2, "gray"), plt.show()
        print()
        print("distance:", str(center_point[1] - bestfit))

    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(
        matchColor=(0, 255, 0),  # draw matches in green color
        singlePointColor=None,
        matchesMask=matchesMask,  # draw only inliers
        flags=2,
    )

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    plt.imshow(img3, "gray"), plt.show()


def ave(values):
    return float(sum(values)) / len(values)


def std(values, ave):
    return math.sqrt(float(sum((value - ave) ** 2 for value in values)) / len(values))


def draw_keypoints(imagename):
    img = cv2.imread(imagename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = sift.detect(gray, None)
    img = cv2.drawKeypoints(gray, kp, img)
    img = cv2.drawKeypoints(gray, kp, img, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite("sift_keypoints.jpg", img)


if __name__ == "__main__":
    main()