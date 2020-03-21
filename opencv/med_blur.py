import numpy as np
import cv2
import glob

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
	save_nozzle = True
	nozzle_frame = None
	timeout = 0
	moving = False

	for imagename in images:
		img = cv2.imread(imagename, 0)
		marked = np.copy(img)

		if prev_image is not None:
			
			result = cv2.absdiff(img,prev_image)
			# binarize image
			# result[result >= 25] = 255
			# result[result < 25] = 0

			nozzle = result[240:280, 354:364]
			pourtest = np.sum(nozzle) > 10000
			
			drip = img[240:280, 354:364]
			mask = np.copy(result)

			#dont count changes in the nozzle area when checking movement
			mask[240:280, 95:625] = 0

			if moving and not pouring:
				prev_frames = []
				cv2.putText(result,"moving", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

			elif not moving and not pouring:
				save_nozzle = True
				cv2.putText(result,"stopped", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
				prev_frames.append(img)
				if pourtest:
					pouring = True
					pourtest = False
					nozzle_frame = prev_frames[0]

			if pouring:
				print("img:", imagename, "pouring:", np.sum(nozzle))
				save_nozzle = True
				cv2.putText(result,"pouring", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
				#draw rectangle for pour
				cv2.rectangle(marked, (354,240), (364,280), 255, 2)
				if pourtest:
					pouring = False
					print("STOPPED POURING")		

			moving = np.sum(mask) > 2100000

			if save_nozzle:
				nozzlename = imagename.replace("med", "sub")
				nozzlename = nozzlename.replace('img', "nozzle")
				cv2.imwrite(nozzlename, nozzle)

			subname = imagename.replace("med", "sub")
			cv2.imwrite(subname, result)


				

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