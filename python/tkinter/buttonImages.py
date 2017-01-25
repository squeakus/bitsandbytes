from Tkinter import *
import subprocess
class GUI:
    def __init__(self, root):

        #add buttons
        self.photo = PhotoImage(file="xxx.1.ppm")
        self.button1 = Button(root,command=lambda name = "xxx.1":self.buttonHandler(name), image=self.photo).grid(row=0)
        self.photo2 = PhotoImage(file="xxx.99.ppm")
        self.button2 = Button(root,command=lambda :self.buttonHandler("x99"), image=self.photo2)
        self.button2.grid(row=1)
        self.button2.bind("<Button-3>",lambda event:self.button_popup(event,"9t9"))

        # create a menu
        self.popup = Menu(root, tearoff=0)
        self.popup.add_command(label="medit",command=lambda:self.show_command("next"))
        self.popup.add_command(label="Analyse",command=lambda:self.show_command("previous"))
        self.popup.add_separator()
        self.popup.add_command(label="Home")

    def buttonHandler(self,name):
        print "the name of the button is",name

    def show_command(self,cmd):
        print "executing command",cmd

    def button_popup(self,event,name):
        print "button popup"
        print "button name",name
        cmd = 'ffmedit xxx.99.mesh'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        process.communicate(self.name)

    def do_popup(self,event):
        # display the popup menu
        try:
            self.popup.tk_popup(event.x_root, event.y_root, 0)
        finally:
            #make sure to release the grab (Tk 8.0a1 only)
            self.popup.grab_release()

root = Tk()
myGUI = GUI(root)
root.bind("<Button-3>", myGUI.do_popup)

root.mainloop()
