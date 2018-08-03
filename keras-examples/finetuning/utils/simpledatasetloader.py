# import the necessary packages
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
	def __init__(self, preprocessors=None):
		# store the image preprocessor
		self.preprocessors = preprocessors

		# if the preprocessors are None, initialize them as an
		# empty list
		if self.preprocessors is None:
			self.preprocessors = []

	def load(self, imagePaths, labelled=True, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []

		# loop over the input images
		for (i, imagePath) in enumerate(imagePaths):
			if imagePath.endswith("FHC.jpg"):
				# load the image and extract the class label assuming
				# that our path has the following format:
				# /path/to/dataset/{class}/{image}.jpg
				image = cv2.imread(imagePath)

				# check to see if our preprocessors are not None
				if self.preprocessors is not None:
					# loop over the preprocessors and apply each to
					# the image
					for p in self.preprocessors:
						image = p.preprocess(image)

				# treat our processed image as a "feature vector"
				# by updating the data list followed by the labels
				data.append(image)
				if labelled:
					label = imagePath.split(os.path.sep)[-2]
					labels.append(label)
				else:
					labels.append(imagePath)

				# show an update every `verbose` images
				if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
					print("[INFO] processed {}/{}".format(i + 1,
						len(imagePaths)))

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))

	def load_image(self, imagePath, labelled=True, verbose=-1):
		# initialize the list of features and labels
		data = []
		labels = []

		# load the image and extract the class label assuming
		# that our path has the following format:
		# /path/to/dataset/{class}/{image}.jpg
		image = cv2.imread(imagePath)

		# check to see if our preprocessors are not None
		if self.preprocessors is not None:
			# loop over the preprocessors and apply each to
			# the image
			for p in self.preprocessors:
				image = p.preprocess(image)

		# treat our processed image as a "feature vector"
		# by updating the data list followed by the labels
		data.append(image)
		if labelled:
			label = imagePath.split(os.path.sep)[-2]
			labels.append(label)

		# return a tuple of the data and labels
		return (np.array(data), np.array(labels))