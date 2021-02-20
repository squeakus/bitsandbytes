import numpy as np
import pandas as pd
import cv2
import glob
from tqdm import tqdm


def main():
    columns = ["area", "perimeter", "circ_x", "circ_y", "radius", "cnt_count"]
    dtype = ["int", "int", "int", "int", "int", "int"]
    bumps = find_bumps("template.jpg", columns, dtype)
    images = glob.glob("*thresh.jpg")
    bumps = parse_layers(images, bumps)

    with open("result.txt", "w") as outfile:
        outfile.write(str(bumps))


def parse_layers(images, bumps, debug=True):
    for imagename in tqdm(images):
        layer_cnt = int(imagename.split("_")[0])
        image = cv2.imread(imagename, 0)
        output = cv2.imread(imagename)
        for bump in bumps:
            x, y, w, h = bump["bbox"]
            layers = bump["layers"]
            layer = layers.iloc[layer_cnt]
            crop = image[y : y + h, x : x + w]
            ret, thresh = cv2.threshold(crop, 127, 255, 0)

            # calc those contours!
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # only compute for maximum contour
            maxarea, maxcnt = 0, 0
            for contour in contours:
                cv2.drawContours(output, [contour], 0, (255, 255, 0), 3)
                area = cv2.contourArea(contour)
                if area > maxarea:
                    maxarea = area
                    maxcnt = contour

            area = cv2.contourArea(maxcnt)
            perimeter = cv2.arcLength(maxcnt, True)
            (circ_x, circ_y), radius = cv2.minEnclosingCircle(maxcnt)

            # update the dataframe
            layer.area = area
            layer.perimeter = perimeter
            layer.circ_x = int(x + circ_x)
            layer.circ_y = int(y + circ_y)
            layer.radius = int(radius)
            layer.cnt_count = len(contours)

        if debug:
            outname = imagename.replace(".jpg", "_out.jpg")

            for bump in bumps:
                layer = bump["layers"].iloc[layer_cnt]
                circ_x = int(layer.circ_x)
                circ_y = int(layer.circ_y)
                radius = int(layer.radius)
                cv2.drawContours(output, [maxcnt], 0, (0, 255, 0), 3)
                output = cv2.circle(output, (circ_x, circ_y), radius, (0, 0, 255), 2)
                output = cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imwrite(outname, output)

    return bumps


def find_bumps(filename, columns, dtypes):
    image = cv2.imread(filename, 0)
    bboxes = []
    bumps = []
    ret, thresh = cv2.threshold(image, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("Could not find any contours")
    else:
        for cnt in contours:

            x, y, w, h = cv2.boundingRect(cnt)
            x = x - 20
            y = y - 20
            w = w + 40
            h = h + 40

            if (w > 80) and (h > 80):
                bboxes.append((x, y, w, h))

    for bbox in bboxes:
        df = pd.DataFrame(np.nan, index=range(0, 193), columns=columns)
        # df.astype(dtypes)
        bump = {"bbox": (x, y, w, h), "layers": df}
        bumps.append(bump)
    return bumps


if __name__ == "__main__":
    main()