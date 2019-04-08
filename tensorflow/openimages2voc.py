import cv2
import xml.etree.cElementTree as ET


def main():
    '''
    Read the labels from the CSV, count em, and write out to a new folder
    ''' 
    train = pull_labels('sub-train-annotations-bbox.csv', 'train')
    test = pull_labels('sub-test-annotations-bbox.csv', 'test')

    count_labels(train)
    count_labels(test)

    write_csv(train, 'train')
    write_csv(test, 'test')


def write_csv(dataset, name):

    with open(name+'/'+name+'.csv', 'w') as csvfile:

        csvfile.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")
        for row in dataset:
            csvfile.write(','.join(map(str, row))+'\n')


def write_VOC():

    annotation = ET.Element("annotation")
    folder = ET.SubElement(annotation, "folder").text = "JPEGImages"
ET.SubElement(annotation, "folder").text = "JPEGImages"


    ET.SubElement(doc, "field1", name="blah").text = "some value1"
    ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"

    tree = ET.ElementTree(root)
    tree.write("filename.xml")

def count_labels(dataset):
    labeldict = {}

    for row in dataset:
        label = row[3]
        if label in labeldict:
            labeldict[label] += 1
        else:
            labeldict[label] = 1
    print(labeldict)


def pull_labels(filename, dataset):
    print("processing dataset:", dataset)

    results = []

    with open(filename, 'r') as infile:

        headers = infile.readline()
        headers = headers.rstrip().split(',')

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

            results.append([filename, width, height, label,
                            xpixmin, xpixmax, ypixmin, ypixmax])

    return results


if __name__ == '__main__':
    main()
