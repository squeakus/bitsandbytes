import numpy as np
import pandas as pd
import cv2
import glob
from tqdm import tqdm
import os
import psutil


def main():
    images = glob.glob("*thresh.jpg")
    process = psutil.Process(os.getpid())

    columns = ["area", "perimeter", "circ_x", "circ_y", "radius", "cnt_count"]
    print(f"before: {round(process.memory_info().rss/ (1024*1024),3)} megabytes")
    bumps = find_bumps("template.jpg", columns)
    # bumps = parse_layers(images, bumps)

    vol = volumize(images)
    print(f"Memory used: {round(process.memory_info().rss/ (1024*1024),3)} megabytes")

    with open("volume.npy", "wb") as f:
        np.save(f, vol)

    # with open("result.txt", "w") as outfile:
    #     outfile.write(str(bumps))


def volumize(images):
    layers = []
    for imagename in tqdm(images):
        image = cv2.imread(imagename, 0)
        ret, thresh = cv2.threshold(image, 127, 255, 0)
        layers.append(thresh)

    vol = np.stack(layers, axis=0)
    print(f"Loaded volume  {vol.shape}  {vol.dtype}")


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
                area = cv2.contourArea(contour)
                if area > maxarea:
                    maxarea = area
                    maxcnt = contour

            bump["contours"] = contours
            bump["maxcnt"] = maxcnt

            if maxarea == 0:
                area, perimeter, circ_x, circ_y, radius = 0, 0, 0, 0, 0
            else:
                area = cv2.contourArea(maxcnt)
                perimeter = cv2.arcLength(maxcnt, True)
                (circ_x, circ_y), radius = cv2.minEnclosingCircle(maxcnt)

            # update the dataframe
            layer.area = area
            layer.perimeter = perimeter
            layer.circ_x = x + circ_x
            layer.circ_y = y + circ_y
            layer.radius = radius
            layer.cnt_count = len(contours)

        if debug:
            outname = imagename.replace(".jpg", "_out.jpg")

            for bump in bumps:
                x, y, w, h = bump["bbox"]
                maxcnt = bump["maxcnt"]
                layer = bump["layers"].iloc[layer_cnt]
                circ_x = layer.circ_x
                circ_y = layer.circ_y
                radius = layer.radius
                if (radius * 2) < w:
                    # output = cv2.circle(output, (circ_x, circ_y), radius, (0, 0, 255), 2)
                    output = cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    # if layer.area > 0:
                    #    for coord in maxcnt:
                    #        coord[0][0] += x
                    #        coord[0][1] += y
                    #    cv2.drawContours(output, [maxcnt], 0, (255, 255, 0), 2)
            cv2.imwrite(outname, output)

    return bumps


def find_bumps(filename, columns):
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
        x, y, w, h = bbox
        df = pd.DataFrame(0, index=range(0, 193), columns=columns, dtype=int)
        # df.astype(dtypes)
        bump = {"bbox": (x, y, w, h), "layers": df}
        bumps.append(bump)
    return bumps


if __name__ == "__main__":
    main()