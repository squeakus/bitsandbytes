import wx

class MyFrame(wx.Frame):

    PENS = ['blue','red','green','black','purple']
    
    def __init__(self,parent=None,id=wx.ID_ANY,title="Line Drawer"):
        wx.Frame.__init__(self, parent, id, title)

        
        self.sizer = wx.FlexGridSizer(2,2,0,0)

        # Add canvas
        self.canvas = wx.ScrolledWindow(self, id=wx.ID_ANY)
        self.canvas.EnableScrolling(True, True)
        self.P_WIDTH = 1000
        self.P_HEIGHT = 1000
        self.canvas.SetScrollbars(20, 20, self.P_WIDTH/20, self.P_HEIGHT/20)
        self.sizer.Add(self.canvas, 1, wx.EXPAND)

        # Pad spacing to sizer
        self.pad_sizer=wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.pad_sizer, 1, wx.EXPAND)

        self.sizer.AddGrowableRow(0, 1)
        self.sizer.AddGrowableCol(0, 1)

        # Add text entry
        self.text_sizer=wx.BoxSizer(wx.HORIZONTAL)
        self.text_sizer.Add(wx.StaticText(self,label="  f(y)=   "),0,wx.ALIGN_CENTER)
        self.entry=wx.TextCtrl(self,style=wx.TE_PROCESS_ENTER)
        self.text_sizer.Add(self.entry,1,0,wx.ALL)
        self.sizer.Add(self.text_sizer,0,wx.EXPAND)

        # Add button
        self.button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.button1=wx.Button(self,label="Show")
        self.button_sizer.Add(self.button1,0,wx.ALIGN_RIGHT)
        self.sizer.Add(self.button_sizer,0,wx.EXPAND)

        self.SetSizer(self.sizer)

        # housekeeping
        self.lines = []
        self.button1.Bind(wx.EVT_BUTTON,self.OnClick,self.button1)
        #self.entry.Bind(wx.EVT_COMMAND_ENTER,self.OnClick,self.entry) # <--- This doesn't work correctly.  TO-DO: make TextCtrl respond to <Enter>.
        self.canvas.Bind(wx.EVT_PAINT, self.OnPaint)
       
        
        
    def OnPaint(self, event):
        dc = wx.PaintDC(self.canvas)

        # This does different colors.
        for line_no in range(len(self.lines)):
            
            dc.SetPen(wx.Pen(self.PENS[line_no % len(self.PENS)],1))
            dc.DrawLine(*self.lines[line_no])

        # If you want all one color, use this instead:
        # dc.SetPen('black',1)
        # for line in self.lines:
        #    dc.DrawLine(*line)
        
    def OnClick(self, event=None):
        text = self.entry.GetValue()
        coords = text.split(',')
        try:
            x0,y0,x1,y1 = [int(x) for x in coords]
        except:
            return
        else:
            self.lines.append((x0,y0,x1,y1))
        self.canvas.Refresh()  # <-- The key to getting the drawing to work.
        
if __name__ == "__main__":

    app = wx.PySimpleApp()  # <-- much easier than MDIParentFrame...
    app.frame = MyFrame()
    app.frame.Show()
    app.MainLoop()
