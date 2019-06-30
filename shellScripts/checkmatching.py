#!/usr/bin/python3
import glob, os


def main():
	images = find("*.jpg")

	for image in images:
		xmlname = image.replace(".jpg", ".xml")
		if not os.path.isfile(xmlname):
			print(image, "has no matching xml file:", xmlname)
		 
	xmlfiles =  find("*.xml")
	for xmlfile in xmlfiles:
		imagename = xmlfile.replace(".xml", ".jpg")
		if not os.path.isfile(xmlname):
			print(xmlfile, "has no matching xml file:", imagename)

def find(regex, folder='./'):
    found = []
    for filename in glob.iglob(folder+'**/'+regex, recursive=True):
        found.append(filename)
    return found


if __name__ == '__main__':
	main()