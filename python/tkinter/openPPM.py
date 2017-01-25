import os
import Tkinter

root = Tkinter.Tk()
L = Tkinter.Listbox(selectmode=Tkinter.BROWSE)
ppmDict ={}

for ppmName in os.listdir(os.getcwd()):
    if ppmName.endswith('5.ppm'):
        print "adding",ppmName
        ppm = Tkinter.PhotoImage(file=ppmName)
        ppmDict[ppmName]= ppm
        L.insert(Tkinter.END, ppmName)
L.pack()
img = Tkinter.Label()
img.pack()
def list_entry_clicked(*ignore):
    imgname = L.get(L.curselection()[0])
    img.config(image=ppmDict[imgname])

L.bind('<ButtonRelease-1>',list_entry_clicked)
root.mainloop()
