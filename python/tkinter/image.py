from Tkinter import *
root = Tk()

photo = PhotoImage(file="xxx.1.ppm")
w = Label(root, image=photo)
#w.photo = photo
w.pack()
root.mainloop()
