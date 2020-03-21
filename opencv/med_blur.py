import numpy as np
import cv2
import glob
import imutils

def main():
	# img = cv2.imread('img00002.jpg', 0)
	# img[240:280, 95:625] = 255
	# cv2.imshow("ROI", img)
	# cv2.waitKey(0)
	subtract_images()

def subtract_images():
	images = find("*.jpg", folder="./med")
	images.sort()
	prev_image = None
	pouring = False
	prev_frames = []
	nozzle_frame = None
	timeout = 0
	moving = False
	imagetxt = ""

	for imagename in images:
		img = cv2.imread(imagename, 0)
		marked = np.copy(img)

		if prev_image is not None:
			# subtract the images
			result = cv2.absdiff(img,prev_image)
			#cut out the nozzle
			nozzlediff= result[240:280, 354:364]
			nozzleout = marked[240:280, 354:364]
			#sum the difference to check for flow
			pourtest = np.sum(nozzlediff) > 10000
			
			drip = img[240:280, 354:364]
			mask = np.copy(result)

			#dont count changes in the nozzle area when checking movement
			mask[240:280, 95:625] = 0

			if moving and not pouring:
				imagetxt = "Moving"
				prev_frames = []
				
			elif not moving and not pouring:
				imagetxt = "Stopped"
				prev_frames.append(img)

				if pourtest:
					print("pour start:", imagename)
					pouring = True
					pourtest = False
					nozzle_frame = prev_frames[0]

			if pouring:
				imagetxt = "Pouring"
				nozzlename = imagename.replace("med", "noz")
				nozzleout = imutils.resize(nozzleout, height=480)
				cv2.imwrite(nozzlename, nozzleout)
				#draw rectangle for pour
				cv2.rectangle(marked, (354,240), (364,280), 255, 2)


				if pourtest:
					pouring = False
					print("pour finish:", imagename)

			moving = np.sum(mask) > 2100000
		cv2.putText(marked, imagetxt, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
		outname = imagename.replace("med", "out")
		cv2.imwrite(outname, marked)

		prev_image = img

def med_blur():
	images = find("*.jpg")
	for imagename in images:
		print("processing image", imagename)
		img = cv2.imread(imagename, 1)

		median_blur= cv2.medianBlur(img, 5)
		cv2.imwrite("med/"+imagename, median_blur)


def find(regex, folder='./'):
    found = []
    for filename in glob.iglob(folder+'**/'+regex, recursive=True):
        found.append(filename)
    return found

if __name__ == "__main__":
	main()