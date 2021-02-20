import numpy as np
import pandas as pd
import cv2
import glob


def main():
    columns = ["area", "perimeter", "circle", "cnt_count"]
    bumps = find_bumps("template.jpg", columns)
    images = glob.glob("*thresh.jpg")
    bumps = parse_layers(images, bumps)

    with open("result.txt", "w") as outfile:
        outfile.write(bumps)


def parse_layers(images, bumps, debug=False):
    for imagename in images:
        layer_cnt = int(imagename.split("_")[0])
        image = cv2.imread(imagename, 0)
        print(f"processing layer {layer_cnt}")

        for bump in bumps:
            x, y, w, h = bump["bbox"]
            layers = bump["layers"]
            layer = layers.iloc[layer_cnt]
            crop = image[y : y + h, x : x + w]
            ret, thresh = cv2.threshold(crop, 127, 255, 0)

            # calc those contours!
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            (circ_x, circ_y), radius = cv2.minEnclosingCircle(cnt)

            # update the dataframe
            layer.area = area
            layer.perimeter = perimeter
            layer.circle = (int(x + circ_x), int(y + circ_y), int(radius))
            layer.cnt_count = len(contours)

        if debug:
            outname = imagename.replace(".jpg", "_out.jpg")
            output = cv2.imread(filename)

            for bump in bumps:
                layer = bump["layers"].iloc[layer_cnt]
                circ_x, circ_y, radius = layer.circle

                cv2.drawContours(output, [cnt], 0, (0, 255, 0), 3)
                output = cv2.circle(output, (circ_x, circ_y), radius, (0, 0, 255), 2)
                output = cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

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
        df = pd.DataFrame(np.nan, index=range(0, 193), columns=columns)
        bump = {"bbox": (x, y, w, h), "layers": df}
        bumps.append(bump)
    return bumps


if __name__ == "__main__":
    main()