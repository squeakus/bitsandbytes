# import the necessary packages
from keras.applications import VGG16, ResNet50, MobileNet

import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=1,
help="whether or not to include top of CNN")
args = vars(ap.parse_args())

# load the VGG16 network
#print("[INFO] loading network...")
#model = VGG16(weights="imagenet", include_top=args["include_top"] > 0)
#print("[INFO] showing layers...")


# load the Resnet 50 network
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=args["include_top"] > 0)
print("[INFO] showing layers...")

# load the mobilenet V2 network
# model = MobileNet(input_shape=(224,224,3), include_top=args["include_top"] > 0, weights='imagenet')


# loop over the layers in the network and display them to the
# console
print("[INFO] network overview...")
print(model.summary())
print("[INFO] showing layers...")
for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))

