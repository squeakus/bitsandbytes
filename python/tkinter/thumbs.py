import os
from Tkinter import *
root = Tk()
myContainer1 = Frame(root)  
myContainer1.pack()        
for ppmName in os.listdir(os.getcwd()):
    if ppmName.endswith('5.ppm'):
        print "adding",ppmName
        image = PhotoImage(file=ppmName)
        w = Label(root, image=image)
        w.pack()

root.mainloop()
