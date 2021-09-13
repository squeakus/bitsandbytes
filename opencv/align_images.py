import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt


def main(image1name, image2name, algo="ORB", debug=False):

    MIN_MATCH_COUNT = 10

    image1 = cv2.imread(image1name)  # queryImage
    image2 = cv2.imread(image2name)  # trainImage
    gray1 = cv2.imread(image1name, 0)  # queryImage
    gray2 = cv2.imread(image2name, 0)  # trainImage
    # Initiate detector

    if algo == "SIFT":
        detector = cv2.SIFT_create()
    elif algo == "ORB":
        detector = cv2.ORB_create()
    else:
        print(f"Algorithm {algo} not recognised!")
        exit()

    # find the keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    if debug:
        out1name = image1name.replace(".jpg", "_keypoints.png")
        out2name = image2name.replace(".jpg", "_keypoints.png")
        out1 = cv2.drawKeypoints(gray1, kp1, image1, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out2 = cv2.drawKeypoints(gray2, kp2, image2, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imwrite(out1name, out1)
        cv2.imwrite(out2name, out2)

    brute_force = cv2.BFMatcher()
    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = brute_force.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = gray1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print(f"Not enough matches are found - {len(good)}/{MIN_MATCH_COUNT}")
        matchesMask = None

    if debug:
        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matchesMask,  # draw only inliers
            flags=2,
        )
        for dmatch in good:
            print(dmatch.imgIdx, dmatch.queryIdx, dmatch.trainIdx, dmatch.distance)

        match_image = cv2.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)

        cv2.imwrite("matched.png", match_image)

    print("image1 keypoints:", len(kp1))
    print("image2 keypoints:", len(kp2))
    print("no. of matches:", len(good))
    print("no. of inliers:", matchesMask.count(1))

    # save the realigned trained image
    # Find homography
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match[0].queryIdx].pt
        points2[i, :] = kp2[match[0].trainIdx].pt

    homography, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = image2.shape
    image2_aligned = cv2.warpPerspective(image2, homography, (width, height))
    aligned1name = image1name.replace(".jpg", "_aligned.png")
    aligned2name = image2name.replace(".jpg", "_aligned.png")

    cv2.imwrite(aligned1name, image1)
    cv2.imwrite(aligned2name, image2_aligned)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print(f"usage: python {sys.argv[0]} <image1name> <image2name>")
