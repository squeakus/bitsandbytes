# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import paths
import numpy as np
import random
import argparse
import imutils
import cv2

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
    ap.add_argument("-i", "--image", default=None,
        help="path to input image")
    ap.add_argument("-d", "--dataset", default=None,
        help="path to multiple images")

    args = vars(ap.parse_args())

    if args['image'] is not None:
        show_img(args["image"], args['model'])
    elif args['dataset'] is not None:
        show_imgs(args["dataset"], args['model'])
    else:
        print("an image or dataset must be specified")
        exit()

def show_imgs(datapath, model):
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(datapath)))
    random.seed(42)
    random.shuffle(imagePaths)

    for imagePath in imagePaths:
        show_img(imagePath, model)


def show_img(imgname, model):
    # load the image
    image = cv2.imread(imgname)
    orig = image.copy()
     
    # pre-process the image for classification
    image = cv2.resize(image, (28, 28))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)


    # load the trained convolutional neural network
    print("[INFO] loading network...")
    model = load_model(model)
     
    # classify the input image
    (notSanta, santa) = model.predict(image)[0]
    print("santa:", percent(santa, 1), "notsanta:", percent(notSanta, 1))

    # build the label
    label = "Santa" if santa > notSanta else "Not Santa"

    proba = santa if santa > notSanta else notSanta
    label = "{}: {:.2f}%".format(label, proba * 100)
     
    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (0, 255, 0), 2)
     
    # show the output image
    show(output)


def percent(partial, total):
    percent = round(100 * ( partial / total),1)
    return percent

def show(image, windowname="image"):
    cv2.imshow(windowname, image)

    k = cv2.waitKey(0)
    if k == ord('q') or k == 27:
        print("quitting")
        cv2.destroyAllWindows()
        exit()
    else:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()