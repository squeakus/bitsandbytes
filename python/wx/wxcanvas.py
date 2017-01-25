# using wxPython's wx.lib.plot.PlotCanvas()
# to show a line graph of some trig functions
# also save graph to an image file

import wx
import wx.lib.plot
import math

class MyFrame(wx.Frame):
    def __init__(self, parent, mytitle, mysize):
        wx.Frame.__init__(self, parent, wx.ID_ANY, mytitle, size=mysize)

        # calculate data lists of (x, y) tuples
        # x is in radians
        sin_data = []
        cos_data = []
        x = 0
        while True:
            y = math.sin(x*x)
            sin_data.append((x, y))
            y = math.sin(x*x) + math.cos(x**0.5)
            cos_data.append((x, y))
            x += 0.01
            if x > 4*math.pi:
                break

        # set up the plotting canvas
        plot_canvas = wx.lib.plot.PlotCanvas(self)
        # get client usable size of frame
        frame_size = self.GetClientSize()
        # needed for SaveFile() later
        #plot_canvas.SetInitialSize(size=frame_size)
        # optional allow scrolling
        plot_canvas.SetShowScrollbars(True)
        # optional drag/draw rubberband area to zoom
        # doubleclick to return to original
        # right click to shrink
        plot_canvas.SetEnableZoom(True)
        # optional
        # set the tick and axis label font size (default is 10 point)
        plot_canvas.SetFontSizeAxis(point=8)
        # set title font size (default is 15 point)
        plot_canvas.SetFontSizeTitle(point=10)

        # connect (x, y) points in data list with a line
        sin_lines = wx.lib.plot.PolyLine(sin_data, colour='red', width=1)
        cos_lines = wx.lib.plot.PolyLine(cos_data, colour='blue', width=1)

        # assign lines, title and axis labels
        gc = wx.lib.plot.PlotGraphics([sin_lines, cos_lines],
            'red=sin(x*x)   blue=sin(x*x) + cos(x**0.5)',
            'X Axis (radians)', 'Y Axis')
        # draw the plot and set x and y axis sizes
        plot_canvas.Draw(gc, xAxis=(0, 7), yAxis=(-2, 2))

        # optional save graph to an image file
        plot_canvas.SaveFile(fileName='trig1.jpg')


app = wx.App(0)
# create a MyFrame instance and show the frame
caption = "wx.lib.plot.PlotCanvas() Line Graph"
MyFrame(None, caption, (400, 300)).Show()
app.MainLoop()
