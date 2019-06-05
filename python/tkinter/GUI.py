import os, subprocess
from tkinter import *

class AutoScrollbar(Scrollbar):
    # a scrollbar that hides itself if it's not needed.  only
    # works if you use the grid geometry manager.
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("grid", "remove", self)
        else:
            self.grid()
        Scrollbar.set(self, lo, hi)
    def pack(self, **kw):
        raise (TclError, "cannot use pack with this widget")
    def place(self, **kw):
        raise (TclError, "cannot use place with this widget")

class GUI:
    def __init__(self,root):
        self.defCol = root.cget("bg")
        self.images={}
        self.buttons={}
        self.width = 4
        self.lastButton = None
        self.rw,self.col=0,0
        
        #make the scrollbar
        self.vscrollbar = AutoScrollbar(root,width=20)
        self.vscrollbar.grid(row=0, column=1, sticky=N+S)
        
        #create canvas
        self.canvas = Canvas(root,height=600,width=880,
                             yscrollcommand=self.vscrollbar.set)
        self.canvas.grid(row=0, column=0, sticky=N+S+E+W)
        self.vscrollbar.config(command=self.canvas.yview)        

        # make the canvas expandable
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # create canvas contents
        self.frame = Frame(self.canvas)
        self.frame.rowconfigure(1, weight=1)
        self.frame.columnconfigure(1, weight=1)

        #make the frame
        for ppmName in os.listdir(os.getcwd()):
            if ppmName.endswith('.ppm'):
                name = ppmName.strip('.ppm')
                print("adding",name)
                self.images[name] = PhotoImage(file=ppmName)
                self.buttons[name] = Button(self.frame,command=lambda 
                                            x=name:self.buttonHandler(x), 
                                            image=self.images[name],
                                            bd=5)
                self.buttons[name].grid(row=self.rw,column=self.col)
                self.buttons[name].bind("<Button-3>",lambda event,x=name:self.right_click(event,x))
                self.col += 1
                if self.col == self.width:
                    self.col= 0
                    self.rw += 1
        
        # create a menu
        self.popup = Menu(root, tearoff=0)
        self.popup.add_command(label="medit",command=lambda:self.run_command("ffmedit"))
        self.popup.add_command(label="Analyse",command=lambda:self.run_command("bmpost"))
        self.popup.add_separator()
        self.popup.add_command(label="Home")


        self.canvas.create_window(0, 0, anchor=NW, window=self.frame)
        self.frame.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def buttonHandler(self,name):
        self.buttons[name].focus_force()
        if self.buttons[name]['background']=="green":
            print("defCol:",self.defCol)
            self.buttons[name]['background']=self.defCol
            self.buttons[name]['relief']="raised"
        else:
            self.buttons[name]['background']="green"
            self.buttons[name]['relief']="sunken"
        print("the name of the button is",name)

    def right_click(self,event,name):
        print("button name",name)
        self.lastButton = name
        self.show_popup(event)
        
    def show_popup(self,event):
        try:
            self.popup.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popup.grab_release()

    def run_command(self,cmd):
        print("executing command ",cmd,"on individual",self.lastButton)
        cmd = cmd+" "+self.lastButton
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        process.communicate()
        
root = Tk()
myGUI = GUI(root)
root.mainloop()
