# script to pull down images from google images
# // pull down jquery into the JavaScript console
# var script = document.createElement('script');
# script.src = "https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js";
# document.getElementsByTagName('head')[0].appendChild(script);

# var urls = $('.rg_di .rg_meta').map(function() { return JSON.parse($(this).text()).ou; });
# document.getElementsByTagName('head')[0].appendChild(script);
# var textToSave = urls.toArray().join('\n');
# var hiddenElement = document.createElement('a');
# hiddenElement.href = 'data:attachment/text,' + encodeURI(textToSave);
# hiddenElement.target = '_blank';
# hiddenElement.download = 'urls.txt';
# hiddenElement.click();


# import the necessary packages
from imutils import paths
import argparse
import requests
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-u", "--urls", required=True,
	help="path to file containing image URLs")
ap.add_argument("-o", "--output", required=True,
	help="path to output directory of images")
args = vars(ap.parse_args())

# grab the list of URLs from the input file, then initialize the
# total number of images downloaded thus far
rows = open(args["urls"]).read().strip().split("\n")
total = 0

# create output folder if it doesnt exist

if not os.path.exists(args["output"]):
    os.makedirs(args["output"])
# loop the URLs
for url in rows:
	try:
		# try to download the image
		r = requests.get(url, timeout=60)
 
		# save the image to disk
		p = os.path.sep.join([args["output"], "{}.jpg".format(
			str(total).zfill(8))])
		f = open(p, "wb")
		f.write(r.content)
		f.close()
 
		# update the counter
		print("[INFO] downloaded: {}".format(p))
		total += 1
 
	# handle if any exceptions are thrown during the download process
	except:
		print("[INFO] error downloading {}...skipping".format(p))

# loop over the image paths we just downloaded
for imagePath in paths.list_images(args["output"]):
	# initialize if the image should be deleted or not
	delete = False
 
	# try to load the image
	try:
		image = cv2.imread(imagePath)
 
		# if the image is `None` then we could not properly load it
		# from disk, so delete it
		if image is None:
			delete = True
 
	# if OpenCV cannot load the image then the image is likely
	# corrupt so we should delete it
	except:
		print("Except")
		delete = True
 
	# check to see if the image should be deleted
	if delete:
		print("[INFO] deleting {}".format(imagePath))
		os.remove(imagePath)