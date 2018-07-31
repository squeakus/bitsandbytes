import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
# borrowed from pyimagesearch code
import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
from utils.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utils.aspectawarepreprocessor import AspectAwarePreprocessor
from utils.simpledatasetloader import SimpleDatasetLoader
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
TL_EPOCHS = 50
FT_EPOCHS = 200

def main():
  """Use transfer learning and fine-tuning to train a network on a new dataset"""
  a = argparse.ArgumentParser()
  a.add_argument("-d", "--dataset", required=True,
                 help="path to input dataset")
  a.add_argument("-m", "--model", required=True, help="output model file")
  a.add_argument("--plot", action="store_true")

  args = a.parse_args()

  if (not os.path.exists(args.dataset)):
    print("directories do not exist")
    sys.exit(1)

  # construct the image generator for data augmentation
  aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

  # grab the list of images that we'll be describing, then extract
  # the class label names from the image paths
  print("[INFO] loading images...")
  imagePaths = list(paths.list_images(args.dataset))
  classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
  classNames = [str(x) for x in np.unique(classNames)]

  # initialize the image preprocessors
  aap = AspectAwarePreprocessor(IM_WIDTH, IM_HEIGHT)
  iap = ImageToArrayPreprocessor()

  # load the dataset from disk then scale the raw pixel intensities to
  # the range [0, 1]
  sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
  (data, labels) = sdl.load(imagePaths, verbose=500)
  data = data.astype("float") / 255.0

  # partition the data into training and testing splits using 75% of
  # the data for training and the remaining 25% for testing
  (trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, random_state=42)

  # convert the labels from integers to vectors
  trainY = LabelBinarizer().fit_transform(trainY)
  testY = LabelBinarizer().fit_transform(testY)

  # setup model
  base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
  model = add_new_last_layer(base_model, len(classNames))

  # transfer learning by turning off all conv layers
  setup_to_transfer_learn(model, base_model)

  # train the head of the network for a few epochs (all other
  # layers are frozen) -- this will allow the new FC layers to
  # start to become initialized with actual "learned" values
  # versus pure random
  print("[INFO] training head...")
  history_tl = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
               validation_data=(testX, testY), epochs=TL_EPOCHS,
               steps_per_epoch=len(trainX) // 32, verbose=1)

  # evaluate the network after initialization
  print("[INFO] evaluating after initialization...")
  predictions = model.predict(testX, batch_size=32)
  print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=classNames))

  #
  plot(history_tl, TL_EPOCHS, "inc_tl_plot.png")
  # fine-tuning
  setup_to_finetune(model)

  # for the changes to the model to take affect we need to recompile
  # the model, this time using SGD with a *very* small learning rate
  print("[INFO] re-compiling model...")
  opt = SGD(lr=0.001)
  model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])

  # train the model again, this time fine-tuning *both* the final set
  # of CONV layers along with our set of FC layers
  print("[INFO] fine-tuning model...")
  history_ft = model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
                                   validation_data=(testX, testY), epochs=FT_EPOCHS,
                                   steps_per_epoch=len(trainX) // 32, verbose=1)
  plot(history_ft, FT_EPOCHS, "inc_ft_plot.png")

  # evaluate the network on the fine-tuned model
  print("[INFO] evaluating after fine-tuning...")
  predictions = model.predict(testX, batch_size=32)
  print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1), target_names=classNames))

  # save the model to disk
  print("[INFO] serializing model...")
  model.save(args.model)

def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  opt = RMSprop(lr=0.001)
  model.compile(optimizer='opt', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet

  Args:
    base_model: keras model excluding top
    nb_classes: # of classes

  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch

  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


def plot(H, epochs, filename="plot.png"):
  # plot the training loss and accuracy
  plt.style.use("ggplot")
  plt.figure()
  N = epochs
  plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
  plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy on " + " ".join(categories))
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc="lower left")
  plt.savefig(filename)


if __name__=="__main__":
  main()


