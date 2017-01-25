import os, subprocess
from Tkinter import *

root = Tk()
images={}
width = 4
rw,col=0,0
buttons={}


def buttonHandler(name):
    print "the name of the button is",name

def right_click(event,name):
        print "button name",name
        cmd = 'ffmedit '+name+'.mesh'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE)
        process.communicate()

idx = 0
for ppmName in os.listdir(os.getcwd()):
    if ppmName.endswith('5.ppm'):
        name = ppmName.strip('.ppm')
        print "adding",name,"idx",idx
        images[idx] = PhotoImage(file=ppmName)
        buttons[idx] = Button(root,command=lambda x=name:buttonHandler(x), image=images[idx])
        buttons[idx].grid(row=rw,column=col)
        buttons[idx].bind("<Button-3>",lambda event,x=name:right_click(event,x))
        col += 1
        idx += 1
        if col == width:
            col= 0
            rw += 1


#root.bind('<Button-1>',leftClick)
root.mainloop()
