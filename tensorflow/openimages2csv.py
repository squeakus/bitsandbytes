import cv2

def main():
	train = pull_labels('sub-train-annotations-bbox.csv', 'train')
	test = pull_labels('sub-test-annotations-bbox.csv', 'test')

	traincsv = open('train/train.csv', 'w')
	traincsv.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")

	for row in test:
		print(str(row))


def pull_labels(filename, dataset):
	results = []

	with open(filename, 'r') as infile:

		headers = infile.readline()
		headers = headers.rstrip().split(',')
		print(headers)

		for line in infile:
			line = line.split(',')
			filename = line[0] + '.jpg'
			label = line[-1].rstrip()
			xmin = float(line[4])
			xmax = float(line[5])
			ymin = float(line[6])
			ymax = float(line[7])

			image = cv2.imread(dataset+'/'+filename)
			height, width, depth = image.shape

			xpixmin = int(xmin * width)
			xpixmax = int(xmax * width)
			ypixmin = int(ymin * height)
			ypixmax = int(ymax * height)

			results.append([filename, label,width, height, xpixmin, xpixmax, ypixmin, ypixmax])
			
	return results


if __name__ == '__main__':
    main()
