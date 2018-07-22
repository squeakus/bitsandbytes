# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from santa_lenet import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

def main():
	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--dataset", required=True,
		help="path to input dataset")
	ap.add_argument("-p", "--plot", type=str, default="plot.png",
		help="path to output accuracy/loss plot")
	args = vars(ap.parse_args())

	# initialize the number of epochs to train for, initial learning rate,
	# and batch size
	EPOCHS = 250
	INIT_LR = 1e-3
	BS = 32
	 
	# initialize the data and labels
	print("[INFO] loading images...")
	data = []
	labels = []
	categories = []
	 
	# grab the image paths and randomly shuffle them
	imagePaths = sorted(list(paths.list_images(args["dataset"])))

	# compute all the categories
	for imagePath in imagePaths:
		label = imagePath.split(os.path.sep)[-2]
		if label not in categories:
			categories.append(label)

	print(categories)
	random.seed(42)
	random.shuffle(imagePaths)

	# loop over the input images
	for imagePath in imagePaths:
		# load the image, pre-process it, and store it in the data list
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (28, 28))
		image = img_to_array(image)
		data.append(image)
	 
		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		for idx, cat in enumerate(categories):
			if cat == label:
				label = idx
		if not isinstance(label, int):
			print("missing category!")
			exit()
		labels.append(label)

	# scale the raw pixel intensities to the range [0, 1]
	print(categories)
	args['model'] = "-".join(categories) + ".model"
	data = np.array(data, dtype="float") / 255.0
	labels = np.array(labels)
	 
	# partition the data into training and testing splits using 75% of
	# the data for training and the remaining 25% for testing
	(trainX, testX, trainY, testY) = train_test_split(data,
		labels, test_size=0.25, random_state=42)
	 
	# convert the labels from integers to vectors
	trainY = to_categorical(trainY, num_classes=len(categories))
	testY = to_categorical(testY, num_classes=len(categories))

	# construct the image generator for data augmentation
	aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
		height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
		horizontal_flip=True, fill_mode="nearest")

	# initialize the model
	print("[INFO] compiling model...")
	print("number of categories", len(categories))
	model = LeNet.build(width=28, height=28, depth=3, classes=len(categories))
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	if len(categories) == 2:
		model.compile(loss="binary_crossentropy", optimizer=opt,
			metrics=["accuracy"])
	else:
		model.compile(loss="categorical_crossentropy", optimizer=opt,
			metrics=["accuracy"])
		

	# train the network
	print("[INFO] training network...")
	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
		validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
		epochs=EPOCHS, verbose=1)
	 
	# save the model to disk
	print("[INFO] serializing network...")
	model.save(args["model"])

	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on " + " ".join(categories))
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(args["plot"])


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