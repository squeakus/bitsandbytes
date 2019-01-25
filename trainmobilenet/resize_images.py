from PIL import Image
import os
from os.path import basename
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("-i", "--image_dir", help="Path to images to be resized", required=True, type=str)
    args = parser.parse_args()
    image_dir = args.image_dir
    out_dir = os.path.join(image_dir, 'resized')
    #create a resize folder
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    resize_images(image_dir, out_dir)


def resize_images(image_dir, out_dir):
    basewidth = 640
    for filename in os.listdir(image_dir):
        filenameOnly, file_extension = os.path.splitext(filename)
        # print (file_extension)
        if (file_extension in [".jpg", '.png']):
            print ("processing", filenameOnly)
            infile = os.path.join(image_dir, filename)
            outfile =  os.path.join(out_dir, filename)
            img = Image.open(infile)
            wpercent = (basewidth/float(img.size[0]))
            if wpercent < 1:
                hsize = int((float(img.size[1])*float(wpercent)))
                img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.save(outfile)
            
    print('Finished resizing')


if __name__  == '__main__':
    main()
