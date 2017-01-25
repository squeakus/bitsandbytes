from Tkinter import *

class MyApp:
    def __init__(self, parent):
        self.myParent = parent
        self.myContainer1 = Frame(parent)
        self.myContainer1.pack()

        #------------------ BUTTON #1 ------------------------------------
        button_name = "OK"

        # command binding
        self.button1 = Button(self.myContainer1,
            command = lambda
            arg1=button_name, arg2=1, arg3="Good stuff!" :
            self.buttonHandler(arg1, arg2, arg3)
            )

        # event binding -- passing the event as an argument
        self.button1.bind("<Return>",
            lambda
            event, arg1=button_name, arg2=1, arg3="Good stuff!" :
            self.buttonHandler_a(event, arg1, arg2, arg3)
            )

        self.button1.configure(text=button_name, background="green")
        self.button1.pack(side=LEFT)
        self.button1.focus_force()  # Put keyboard focus on button1

        #------------------ BUTTON #2 ------------------------------------
        button_name = "Cancel"

        # command binding
        self.button2 = Button(self.myContainer1,
            command = lambda
            arg1=button_name, arg2=2, arg3="Bad  stuff!":
            self.buttonHandler(arg1, arg2, arg3)
            )

        # event binding -- without passing the event as an argument
        self.button2.bind("<Return>",
            lambda
            event, arg1=button_name, arg2=2, arg3="Bad  stuff!" :
            self.buttonHandler(arg1, arg2, arg3)
            )

        self.button2.configure(text=button_name, background="red")
        self.button2.pack(side=LEFT)


    def buttonHandler(self, argument1, argument2, argument3):
        print "    buttonHandler routine received arguments:" \
            , argument1.ljust(8), argument2, argument3

    def buttonHandler_a(self, event, argument1, argument2, argument3):
        print "buttonHandler_a received event", event
        self.buttonHandler(argument1, argument2, argument3)


print "\n"*100 # clear the screen
print "Starting program tt078."
root = Tk()
myapp = MyApp(root)
print "Ready to start executing the event loop."
root.mainloop()
print "Finished       executing the event loop."

