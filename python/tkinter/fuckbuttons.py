from Tkinter import *
root = Tk()
button = Button(root, text="Click Me")
print "Button() returns", button
grid = button.grid(column=4, row=10)
print ".grid() returns", grid
#self.click2 = button
