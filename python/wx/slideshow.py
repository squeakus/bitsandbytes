# create a simple image slide show using the
# wx.PaintDC surface as a canvas and
# DrawBitmap(bitmap, x, y, bool transparent)
# Source:  vegaseat

import wx
import os

class MyFrame(wx.Frame):
    def __init__(self, parent, mysize):
        wx.Frame.__init__(self, parent, wx.ID_ANY, size=mysize)
        self.SetBackgroundColour('brown')
        
        # milliseconds per frame
        self.delay = 2200
        # number of loops
        self.loops = 2 
        # pick sequenced image files you have in the given directory
        # can be a .jpg .png .gif or .bmp image files
        # let's pick just .jpg files here
        image_type = ".jpg"
        image_dir = os.getcwd()  # use working directory for testing
        #image_dir = "./Slides"
        file_list = []
        self.name_list = []
        for file in os.listdir(image_dir):
            path = os.path.join(image_dir, file)
            if os.path.isfile(path) and path.endswith(image_type):
                # just the file name
                self.name_list.append(file)
                # full path name
                file_list.append(path)
        # create a list of image objects
        self.image_list = []
        for image_file in file_list:
            self.image_list.append(wx.Bitmap(image_file))
        
        # bind the panel to the paint event
        wx.EVT_PAINT(self, self.onPaint)

    def onPaint(self, event=None):
        # this is the wxPython drawing surface/canvas
        dc = wx.PaintDC(self)
        while self.loops:
            self.loops -= 1
            for ix, bmp in enumerate(self.image_list):
                # optionally show some image information
                w, h = bmp.GetSize()
                info = "%s  %dx%d" % (self.name_list[ix], w, h)
                self.SetTitle(info)
                # draw the image
                dc.DrawBitmap(bmp, 10, 10, True)
                wx.MilliSleep(self.delay)
                # don't clear on fast slide shows to avoid flicker
                if self.delay > 200:
                    dc.Clear()


app = wx.App()
width = 670
height = 540
MyFrame(None, (width, height)).Show()
app.MainLoop()
