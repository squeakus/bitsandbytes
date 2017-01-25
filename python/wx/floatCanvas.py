import random
import wx
from wx.lib.floatcanvas import FloatCanvas as FC

class MainFrame(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title, size=(800, 600))
        self.toolbar = wx.Panel(self, style=wx.BORDER_DOUBLE)
        self.CreateStatusBar()
        self.CanvasPanel = self.CreateCanvasPanel(self)
        self.Canvas.Bind(FC.EVT_MOTION, self.OnMove)
        self.Canvas.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        box = wx.BoxSizer(wx.VERTICAL)
        box.Add(self.toolbar, 1, wx.EXPAND)
        box.Add(self.CanvasPanel, 20, wx.EXPAND)
        self.SetSizer(box)
        self.Centre()

    def OnLeftUp(self, evt):
        Canvas = self.Canvas
        width, height = Canvas.GetSizeTuple()
        x1 = random.randint(0, width)
        x2 = random.randint(0, width)
        y1 = random.randint(0, height)
        y2 = random.randint(0, height)
        line = [[x1, y1], [x2, y2]]
        self.OnDraw(line)

    def OnDraw(self, segment):
        Canvas = self.Canvas
        Canvas.AddLine(Canvas.PixelToWorld(segment), LineColor="Cyan")
        Canvas.Draw()

    def CreateCanvasPanel(self, parent):
        panel = wx.Panel(parent)
        panel.SetBackgroundColour("black")
        self.Canvas = FC.FloatCanvas(panel, BackgroundColor = "Black")
        self.CanvasBox = wx.BoxSizer(wx.VERTICAL)
        self.CanvasBox.Add(self.Canvas, 1, wx.EXPAND)
        panel.SetSizer(self.CanvasBox)
        return panel

    def OnMove(self, evt):
        """
        Updates the status bar
        """
        Canvas = self.Canvas
        worldCoord = evt.Coords                         #world coordinates and pixel coordinates
        pixelCoord = Canvas.WorldToPixel(worldCoord)    #use what you want
        self.SetStatusText("%.2f, %.2f"%tuple(worldCoord))

class App(wx.App):
    def OnInit(self):
        frame = MainFrame(None, -1, "test")
        frame.Show(True)
        return True

if __name__ == "__main__":
    app = App(redirect=False)  #put True if you want messages being redirected to their own window
    app.MainLoop()
