import wx, time

app = wx.App(False)

frame = wx.Frame(None, title="Draw on Panel")
panel = wx.Panel(frame)

def on_paint(event):
    dc = wx.PaintDC(event.GetEventObject())
    dc.Clear()
    dc.SetPen(wx.Pen("BLACK", 4))
    dc.DrawLine(0, 0, 50, 50)
    dc.DrawCircle(20, 100, 1)
#    dc.CrossHair(50, 70)
    x, y = 50, 50
    for i in range(10):
        dc.Clear()
        time.sleep(0.5)
        dc.DrawCircle(x+i, y+i, 2)

panel.Bind(wx.EVT_PAINT, on_paint)

frame.Show(True)
app.MainLoop()
