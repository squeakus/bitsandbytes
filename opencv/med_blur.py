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
	save_dripdiff = True
	dripdiff_frame = None
	timeout = 0

	for imagename in images:
		img = cv2.imread(imagename, 0)
		marked = np.copy(img)

		if prev_image is not None:
			
			result = cv2.absdiff(img,prev_image)
			# binarize image
			# result[result >= 25] = 255
			# result[result < 25] = 0

			dripdiff = result[240:280, 354:364]
			drip = img[240:280, 354:364]
			mask = np.copy(result)

			#dont count changes in the nozzle area when checking movement
			mask[240:280, 95:625] = 0

			movement = np.sum(mask)
			if movement > 2100000 and not pouring:
				prev_frames = []
				cv2.putText(result,"moving", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)

			elif movement > 2100000 and pouring:
					if timeout < 30:
						save_dripdiff = True
						timeout += 1
					else:
						save_dripdiff = False
						timeout = 0
						pouring = False

			elif movement <= 2100000 and not pouring:
				prev_frames.append(img)
				if np.sum(dripdiff) > 10000:
					pouring = True
					dripdiff_frame = prev_frames[0]

			elif movement <= 2100000 and pouring:
				save_dripdiff = True
				cv2.putText(result,"stopped", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
			print(imagename, "pouring:", pouring, "dripdiff:", save_dripdiff)
			if save_dripdiff:
				dripdiffname = imagename.replace("med", "sub")
				dripdiffname = dripdiffname.replace('img', "dripdiff")
				print(dripdiffname, np.sum(dripdiff))
				cv2.imwrite(dripdiffname, dripdiff)

			subname = imagename.replace("med", "sub")
			cv2.imwrite(subname, result)

			if pouring:
				#draw rectangle for pour
				cv2.rectangle(marked, (354,240), (364,280), 255, 2)
				

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