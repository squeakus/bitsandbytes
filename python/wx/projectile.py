# using wxPython for plotting projectile motion
# draw line graph for 2 different firing angles
# tested with Python25 and wxPython28  vegaseat   26mar2008

import wx
import wx.lib.plot as plot
import math

class MyFrame(wx.Frame):
    def __init__(self):
        self.frame1 = wx.Frame(None, title="wx.lib.plot", id=-1, size=(410, 340))
        self.panel1 = wx.Panel(self.frame1)
        self.panel1.SetBackgroundColour("yellow")
        # set up the plot canvas
        plotter = plot.PlotCanvas(self.panel1)
        #plotter.SetInitialSize(size=(400, 300))
        # enable the zoom feature (drag a box around area of interest)
        plotter.SetEnableZoom(True)
        
        # projectile motion equations:
        # height = y(t) = h0 + (t * v * sin(a)) - (g * t*t)/2
        # range  = x(t) = v * cos(a) * t
        # where:
        # v is the muzzle velocity of the projectile (meter/second)
        v = 100
        # a is the firing angle with repsect to ground (radians)
        # h0 is starting height with respect to ground (meters)
        h0 = 0
        # g is the gravitational pull (meters/second_square)
        g = 9.8

        # now calculate the list of (x, y) data point tuples ...
        # x axis is range (distance) in meters
        # y axis is height in meters
        # first for a firing angle of 45 degrees
        d = 45
        a = math.radians(d)  # gives radians
        data45 = []
        for t in range(0, 200):
            # use the time in increments of tx (0.1) seconds
            tx = t/10.0
            # now calculate the height y
            y = h0 + (tx * v * math.sin(a)) - (g * tx * tx)/2
            # calculate the range x
            x = v * math.cos(a) * tx
            # append the (x, y) tple to the list
            data45.append((x, y))

        # now for a firing angle of 30 degrees
        d = 30
        a = math.radians(d)  # gives radians
        data30 = []
        for t in range(0, 200):
            # use the time in increments of tx (0.1) seconds
            tx = t/10.0
            # now calculate the height y
            y = h0 + (tx * v * math.sin(a)) - (g * tx * tx)/2
            # calculate the range x
            x = v * math.cos(a) * tx
            # append the (x, y) tuple to the data list
            data30.append((x, y))

        # draw points as a line
        # 2 different lines one for an angle of 45 one of 30 degrees
        line45 = plot.PolyLine(data45, colour='red', width=1)
        line30 = plot.PolyLine(data30, colour='blue', width=1)
        # set up the plot
        gc = plot.PlotGraphics([line45, line30],
            'Projetile Motion',
            'range (meters) firing angle red=45, blue=30',
            'height (meters)')
        # and draw it
        plotter.Draw(gc, xAxis=(0,1200), yAxis=(0,300))
        
        self.frame1.Show(True)


app = wx.PySimpleApp()
f = MyFrame()
app.MainLoop()
