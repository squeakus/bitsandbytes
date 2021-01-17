# import the necessary packages
from imutils import paths
import numpy as np
import cv2
import argparse
import pickle
from sklearn.ensemble import IsolationForest


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--train", action="store_true", help="train a new model")
    ap.add_argument("-d", "--dataset", default=None, help="folder of training data")
    ap.add_argument("-m", "--model", required=True, help="anomaly detection model")
    ap.add_argument("-i", "--image", help="input image to evaluate")
    args = vars(ap.parse_args())

    if args["train"]:
        if args["dataset"] is None:
            ap.error("You need to specify a dataset folder (-d) when training")
            exit()
        else:
            print("training a new model")
            train_model(args["dataset"], args["model"])
    else:
        check_image(args["image"], args["model"])


def train_model(dataset, modelname):
    # load and quantify our image dataset
    print("[INFO] preparing dataset...")
    data = load_dataset(dataset, bins=(3, 3, 3))
    # train the anomaly detection model
    print("[INFO] fitting anomaly detection model...")
    model = IsolationForest(n_estimators=1000, contamination=0.01, random_state=42)
    model.fit(data)

    # serialize the anomaly detection model to disk
    f = open(modelname, "wb")
    f.write(pickle.dumps(model))
    f.close()


def check_image(imagename, modelname):
    # load the anomaly detection model
    print("[INFO] loading anomaly detection model...")
    model = pickle.loads(open(modelname, "rb").read())
    # load the input image, convert it to the HSV color space, and
    # quantify the image in the *same manner* as we did during training
    image = cv2.imread(imagename)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    features = quantify_image(hsv, bins=(3, 3, 3))

    # use the anomaly detector model and extracted features to determine
    # if the example image is an anomaly or not
    preds = model.predict([features])[0]
    label = "anomaly" if preds == -1 else "normal"
    color = (0, 0, 255) if preds == -1 else (0, 255, 0)
    # draw the predicted label text on the original image
    cv2.putText(image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # display the image
    cv2.imshow("Output", image)
    cv2.waitKey(0)


def quantify_image(image, bins=(4, 6, 3)):
    # compute a 3D color histogram over the image and normalize it
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # return the histogram
    return hist


def load_dataset(datasetPath, bins):
    # grab the paths to all images in our dataset directory, then
    # initialize our lists of images
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    # loop over the image paths
    for imagePath in imagePaths:
        # load the image and convert it to the HSV color space
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # quantify the image and update the data list
        features = quantify_image(image, bins)
        data.append(features)
    # return our data list as a NumPy array
    return np.array(data)


if __name__ == "__main__":
    main()
