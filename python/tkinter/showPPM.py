import os
from Tkinter import *

root = Tk()
L = Listbox(selectmode=BROWSE)
ppmDict ={}

for ppmName in os.listdir(os.getcwd()):
    if ppmName.endswith('5.ppm'):
        print "adding",ppmName
        image = PhotoImage(file=ppmName)
        ppmDict[ppmName]=image
        w = Label(root, image=image)
        w.pack()
        L.insert(END, ppmName)
L.pack()
img = Label()
img.pack()
def list_entry_clicked(*ignore):
    imgname = L.get(L.curselection()[0])
    img.config(image=ppmDict[imgname])

L.bind('<ButtonRelease-1>',list_entry_clicked)
root.mainloop()
