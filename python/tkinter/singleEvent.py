from Tkinter import *
class MyApp:
    def __init__(self, parent):
        self.myParent = parent
        self.myContainer1 = Frame(parent)
        self.myContainer1.pack()

        button_name = "OK"
        self.button1 = Button(self.myContainer1,
            command=self.buttonHandler(button_name, 1, "Good stuff!"))

        # self.button1.bind("<Return>", self.buttonHandler_a(event, button_name, 1, "Good stuff!"))
        self.button1.configure(text=button_name, background="green")
        self.button1.pack(side=LEFT)
        self.button1.focus_force()  # Put keyboard focus on button1

        button_name = "Cancel"
        self.button2 = Button(self.myContainer1,
            command=self.buttonHandler(button_name, 2, "Bad  stuff!"))

        # self.button2.bind("<Return>", self.buttonHandler_a(event, button_name, 2, "Bad  stuff!"))
        self.button2.configure(text=button_name, background="red")
        self.button2.pack(side=LEFT)


    def buttonHandler(self, arg1, arg2, arg3):
        print "    buttonHandler routine received arguments:", arg1.ljust(8), arg2, arg3

    def buttonHandler_a(self, event, arg1, arg2, arg3):
        print "buttonHandler_a received event", event
        self.buttonHandler(arg1, arg2, arg3)

print "\n"*100 # clear the screen
print "Starting program tt077."
root = Tk()
myapp = MyApp(root)
print "Ready to start executing the event loop."
root.mainloop()
print "Finished       executing the event loop."
