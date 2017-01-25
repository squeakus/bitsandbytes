# create a background image on a wxPython panel
# and place a button on top of the image

import wx
import random

class MyPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, style=wx.RAISED_BORDER)
        
        # you can load .jpg .png .bmp or .gif files
        # pick an image file you have
        # if it's not in the working folder, give the full path
        image_file = 'roses.jpg'
        print "fine1"
        bmp = wx.Bitmap(image_file)
        # image's upper left corner anchors at panel coordinates (0, 0)
        self.bitmap = wx.StaticBitmap(self, wx.ID_ANY, bmp, (0, 0))
        print "fine2"
        # size the panel to fit image
        self.SetSize((bmp.GetWidth(), bmp.GetHeight()))
        # fit the frame around the panel
        parent.Fit()
        print "fine3"
        # button goes on the image --> self.bitmap is the parent
        self.button = wx.Button(self.bitmap, wx.ID_ANY,
            label='Click the button', pos=(10, 10), size=(260, 25))
        # bind the button click event to an action
        print "fine4"
        self.button.Bind(wx.EVT_BUTTON, self.button_click)
        print "fine5"

    def button_click(self, event):
        """button has been clicked, do something"""
        wisdom_list = [
        "Man who run in front of car, get tired.",
        "Man who run behind car, get exhausted.",
        "Man who drive like hell, bound to get there.",
        "Man who scratches butt should not bite fingernails.",
        "Man who fight wife all day, get no piece at night.",
        "Man who sit on tack, get point."]
        self.button.SetLabel(random.choice(wisdom_list))
        # wait 2.5 seconds
        wx.Sleep(2.5)
        self.button.SetLabel("Click the button")


app = wx.App()
# give the frame a titlebar
mytitle = "using an image as a backgroound"
# create the wx.Frame class instance
# the image will set the size
frame = wx.Frame(None, title=mytitle)
# create the MyPanel class instance
MyPanel(frame)
frame.Show(True)
app.MainLoop()

