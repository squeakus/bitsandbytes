'''
A classifier tool that will move folders to relevand class
the last labelled image is stored in checkindex.txt and any revisions to the labels
are stored in revisions.txt
'''
import os
import sys
import glob
import datetime
import time
import json
from shutil import copyfile
from tkinter import Tk, Label, Button, W, E
from PIL import Image, ImageTk


class LabelUI():
    """Interface to label images."""

    def __init__(self, img_dir):
        with open('config.json') as json_file:
            settings = json.load(json_file)

        self.images = []
        # Size of individual image windows
        self.imgwidth = settings['image_width']
        self.imgheight = settings['image_height']
        self.imagecnt = 0
        self.imagename = ""
        self.outfolder = 'labelled_images'
        self.labelled = {}
        self.classes = {}
        self.keypress = settings['keypress']

        # creating a counter dict for each class
        for label in settings['classes']:
            self.classes[label] = 0

        # make all the relevant folders for output
        if not os.path.isdir(self.outfolder):
            os.makedirs(self.outfolder)

        for item in self.classes:
            classpath = os.path.join(self.outfolder, item)
            if not os.path.isdir(classpath):
                os.makedirs(classpath)

        # make a list of all the images
        imagenames = []
        search = os.path.join(img_dir, '*.jpg')
        for filename in glob.glob(search):
            imagenames.append(filename)
        if len(imagenames) == 0:
            print("could not find any images in", search)
            exit()

        self.images = sorted(imagenames)

        # get index from file or create new one
        if os.path.isfile('index.txt'):
            with open('index.txt', 'r') as f:
                self.imagename = f.readline()
                print("starting from previous checkpoint:", self.imagename)
        else:
            self.update_index()

        # find index of last classified image
        for idx, image in enumerate(self.images):
            if image == self.imagename:
                self.imagecnt = idx

        print("Total number of images:", len(self.images))
        newimg = self.images[self.imagecnt]
        self.update_index()

        # make a list of all the labelled images
        search = os.path.join(self.outfolder, '**', '*.jpg')

        for filename in glob.iglob(search, recursive=True):
            filename = filename.split(os.sep)
            category = filename[-2]
            imagename = filename[-1]
            self.labelled[imagename] = category
        print("Total number labelled:", len(self.labelled))

        # set up the GUI and buttons
        self.root = Tk()
        self.root.title("Image Labeller")
        opt = self.open_img(newimg)

        self.optpanel = Label(self.root, image=opt)
        self.optpanel.grid(row=0, columnspan=2)
        current_label = self.get_current_label()
        self.labelpanel = Label(
            self.root, text="Current label: " + current_label)
        self.labelpanel.grid(row=2, columnspan=4)

        row = 3
        col = 0

        # Create buttons for all the labels
        for label in self.classes:
            keypress = None
            for key in self.keypress:
                if self.keypress[key] == label:
                    keypress = key 
            # The label on the button should show the keypress shortcut
            if keypress is None:
                button = Button(self.root, text=label,
                                command=self.create_callback(label))
            else:
                button = Button(self.root, text=label+" ("+keypress+")",
                                command=self.create_callback(label))
            button.grid(row=row, column=col, sticky=W+E)
            # Update the row and col for the next button
            col += 1
            if col % 4 == 0:
                col = 0
                row += 1
        button = Button(self.root,
                        text='redo previous classification (backspace)',
                        command=lambda: self.undo())
        button.grid(row=row+1, column=0, columnspan=4, sticky=W+E)

        self.root.bind('<Key>', self.key_pressed)
        self.root.mainloop()

    def create_callback(self, m):
        return lambda: self.label_img(m)

    def key_pressed(self, event):
        """Map keypress events to functions."""
        if event.keysym == 'Right':
            self.next_image()
        elif event.keysym == ('Left'):
            self.prev_image()
        elif event.keysym == ('BackSpace'):
            self.undo()
        elif event.keysym == ('q'):
            print('quitting')
            exit()
        elif event.char in self.keypress is not None:
            self.label_img(self.keypress[event.char])
        else:
            print("unknown key input")

    def get_current_label(self):
        """Get the assigned label for a given image."""
        currentimage = self.images[self.imagecnt].split(os.sep)[-1]

        if currentimage in self.labelled:
            classification = self.labelled[currentimage]
        else:
            classification = "None"
        return classification

    def update_index(self):
        """
        Writes index to a file so you're not starting
        from scratch each time
        """
        print("current image:", self.imagecnt)
        with open('index.txt', 'w') as f:
            f.write(str(self.images[self.imagecnt]))
            f.close()

    def record_change(self, image, oldlabel, newlabel):
        """
        Makes a record of any renaming to an existing label
        """
        ts = time.time()
        timestamp = datetime.datetime.fromtimestamp(ts)
        outstring = timestamp.strftime('%Y-%m-%d_%H:%M:%S')
        outstring += " " + image + " " + oldlabel + "->" + newlabel + "\n"
        with open('revisions.txt', 'a') as f:
            f.write(outstring)
            f.close()

    def open_img(self, filename):
        """Check if image exists and then open it and resize."""
        if os.path.isfile(filename):
            image = Image.open(filename)
            image = image.resize(
                (self.imgwidth, self.imgheight), Image.ANTIALIAS)
        else:
            print("missing file!", filename)
            image = Image.new('RGB', (self.imgwidth, self.imgheight))

        return ImageTk.PhotoImage(image)

    def undo(self):
        """Go back an image and remove the label"""
        self.imagecnt -= 1
        fullname = self.images[self.imagecnt]
        filename = fullname.split(os.sep)[-1]
        print("undoing label for", filename)
        self.delete_label(filename)
        self.imagecnt += 1
        self.prev_image()

    def delete_label(self, filename):
        """Remove the existing label and delete files in class folder"""
        current_label = self.get_current_label()
        current = filename + "*.jpg"
        search = os.path.join(self.outfolder, current_label, current)
        files = glob.glob(search)
        for fname in files:
            os.remove(fname)
        self.labelled[filename] = "None"

    def prev_image(self):
        """ Go back an image"""
        if self.imagecnt > 0:
            self.imagecnt -= 2
            self.next_image()

    def next_image(self):
        """ Move onto next image"""
        self.imagecnt += 1
        newimg = self.images[self.imagecnt]
        self.update_index()

        opt = self.open_img(newimg)
        self.optpanel.configure(image=opt)
        self.optpanel.image = opt
        
        current_label = self.get_current_label()
        self.labelpanel.configure(text=current_label)
        self.labelpanel.text = "Label: " + current_label

    def label_img(self, label):
        """
        Check if image is labelled, record change
        if it is and then copy it to the relevant
        class subfolder
        """
        current_label = self.get_current_label()
        fullname = self.images[self.imagecnt]
        filename = fullname.split(os.sep)[-1]

        # check if file being renamed or new label
        if current_label == "None":
            print("labelling image as:", label)
        else:
            print("relabelling image:", label)
            self.delete_label(filename)
            self.record_change(filename, current_label, label)

        self.labelled[filename] = label

        img = self.images[self.imagecnt]
        targetname = os.path.join(self.outfolder, label)
        original = self.images[self.imagecnt].split(os.sep)[-1]
        imgtarget = os.path.join(targetname, original)

        copyfile(img, imgtarget)
        self.next_image()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        LabelUI(os.getcwd())
    else:
        LabelUI(sys.argv[1])
