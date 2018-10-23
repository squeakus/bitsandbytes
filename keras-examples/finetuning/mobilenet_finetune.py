# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import MobileNet
import keras.layers as klayers
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from imutils import paths
import matplotlib.pyplot as plt
from utils.imagetoarraypreprocessor import ImageToArrayPreprocessor
from utils.aspectawarepreprocessor import AspectAwarePreprocessor
from utils.simpledatasetloader import SimpleDatasetLoader
import numpy as np
import argparse
import os

IM_WIDTH, IM_HEIGHT = 224, 224
TL_EPOCHS = 25
FT_EPOCHS = 100

def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset")
    ap.add_argument("-m", "--model", required=True,
        help="path to output model")
    args = vars(ap.parse_args())

    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

    trainX, testX, trainY, testY, classNames = process_folder(args)

    # load the mobilenet network, ensuring the head FC layer sets are left
    # off
    baseModel =MobileNet(input_shape=(224,224,3), include_top=False, weights='imagenet')

    # place the head FC model on top of the base model -- this will
    # become the actual model we will train
    model = add_new_fc_layer(baseModel, len(classNames), 256)

    # loop over all layers in the base model and freeze them so they
    # will *not* be updated during the training process
    model = freeze_for_transfer(model, baseModel)


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

    plot(history_tl, TL_EPOCHS, "vgg_tl_plot.png")
    # now that the head FC layers have been trained/initialized, lets
    # unfreeze the final set of CONV layers and make them trainable
    for layer in baseModel.layers[15:]:
        layer.trainable = True

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
    
    # evaluate the network on the fine-tuned model
    print("[INFO] evaluating after fine-tuning...")
    predictions = model.predict(testX, batch_size=32)
    print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=classNames))
    plot(history_ft, FT_EPOCHS, "vgg_ft_plot.png")
    # save the model to disk
    print("[INFO] serializing model...")
    model.save(args["model"])

def process_folder(args):
    # grab the list of images that we'll be describing, then extract
    # the class label names from the image paths
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(args["dataset"]))
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

    return trainX, testX, trainY, testY, classNames

def add_new_fc_layer(baseModel, classes, D):
    # initialize the head model that will be placed on top of
    # the base, then add a FC layer
    alpha = 1
    shape = (1, 1, int(1024 * alpha))
    x = baseModel.output
    x = klayers.GlobalAveragePooling2D()(x)
    x = klayers.Reshape(shape, name='reshape_1')(x)
    x = klayers.Dropout(0.5, name='dropout_1')(x)
    x = klayers.Conv2D(classes, (1, 1),
                      padding='same',
                      name='conv_preds')(x)
    x = klayers.Activation('softmax', name='act_softmax')(x)
    x = klayers.Reshape((classes,), name='reshape_2')(x)
    model = Model(inputs=baseModel.input, outputs=x)
    return model

    # headModel = baseModel.output
    # headModel = GlobalAveragePooling2D()(headModel)
    # headModel.reshape()
    # headModel = Dropout(0.5)(headModel)
    # headModel = Dense(classes, activation="softmax")(headModel) # softmax
    # model = Model(inputs=baseModel.input, outputs=headModel)
    # return the model
    # loop over the layers in the network and display them to the
    # console
    return model

def print_model(model):
    print("[INFO] network overview...")
    print(model.summary())
    print("[INFO] showing layers...")
    for (i, layer) in enumerate(model.layers):
        print("[INFO] {}\t{}".format(i, layer.__class__.__name__))

def freeze_for_transfer(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  
  opt = RMSprop(lr=0.001)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

  return model

def plot(H, epochs, filename="plot.png"):
  # plot the training loss and accuracy
  plt.style.use("ggplot")
  plt.figure()
  N = epochs
  plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
  plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
  plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
  plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
  plt.title("Training Loss and Accuracy")
  plt.xlabel("Epoch #")
  plt.ylabel("Loss/Accuracy")
  plt.legend(loc="lower left")
  plt.savefig(filename)



if __name__ == "__main__":
    main()
