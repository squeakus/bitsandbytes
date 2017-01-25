#!/usr/bin/python

# focusevent.py

import wx


class MyWindow(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        image_file = 'pylon.ppm'
        raw_image = wx.Image(image_file, wx.BITMAP_TYPE_ANY).ShrinkBy(6,6)
        bit_map = raw_image.ConvertToBitmap()
        self.color = '#b3b3b3'
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_SET_FOCUS, self.OnSetFocus)
        self.Bind(wx.EVT_KILL_FOCUS, self.OnKillFocus)
        wx.StaticBitmap(self, -1, bit_map, (0, 0))

    def OnPaint(self, event):
        dc = wx.PaintDC(self)

        dc.SetPen(wx.Pen(self.color))
        x, y = self.GetSize()
        dc.DrawRectangle(0, 0, x, y)

    def OnSize(self, event):
        self.Refresh()

    def OnSetFocus(self, event):
        self.color = '#0099f7'
        self.color = '#003300'
        self.Refresh()

    def OnKillFocus(self, event):
        self.color = '#b3b3b3'
        self.Refresh()

class FocusEvent(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title, size=(350, 350))
        grid = wx.GridSizer(2, 2, 10, 10)
        print "top",wx.TOP ," bottom",wx.BOTTOM ,"left",wx.LEFT,"right",wx.RIGHT
        grid.AddMany([(MyWindow(self), 1, wx.EXPAND|wx.TOP|wx.LEFT,9),
                      (MyWindow(self), 1, wx.EXPAND|wx.TOP|wx.RIGHT, 9), 
                      (MyWindow(self), 1, wx.EXPAND|wx.BOTTOM|wx.LEFT, 9), 
                      (MyWindow(self), 1, wx.EXPAND|wx.BOTTOM|wx.RIGHT, 9),
                      (MyWindow(self), 1, wx.EXPAND|wx.BOTTOM|wx.LEFT, 9),
                      (MyWindow(self), 1, wx.EXPAND|wx.BOTTOM|wx.RIGHT, 9)])


        self.SetSizer(grid)
        self.Centre()
        self.Show(True)

app = wx.App()
FocusEvent(None, -1, 'focus event')
app.MainLoop()
