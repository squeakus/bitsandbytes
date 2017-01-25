from Tkinter import *
class MyApp:
    def __init__(self, parent):
        self.myParent = parent
        self.myContainer1 = Frame(parent)
        self.myContainer1.pack()

        self.button1 = Button(self.myContainer1, command=self.button1Click) ### (1)
        self.button1.configure(text="OK", background= "green")
        self.button1.pack(side=LEFT)
        self.button1.focus_force()


        self.button2 = Button(self.myContainer1, command=self.button2Click)  ### (2)
        self.button2.configure(text="Cancel", background="red")
        self.button2.pack(side=RIGHT)


    def button1Click(self):  ### (3)
        print "button1Click event handler"
        if self.button1["background"] == "green":
            self.button1["background"] = "yellow"
        else:
            self.button1["background"] = "green"

    def button2Click(self): ### (4)
        print "button2Click event handler"
        self.myParent.destroy()



root = Tk()
myapp = MyApp(root)
root.mainloop()
