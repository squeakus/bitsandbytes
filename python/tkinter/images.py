from Tkinter import *
root = Tk()

photo = PhotoImage(file="xxx.1.ppm")
w = Label(root, image=photo).grid(row=0)
photo = PhotoImage(file="xxx.99.ppm")
w = Label(root, image=photo).grid(row=1)
w = Label(root, image=photo).grid(row=0,column=1)
w = Label(root, image=photo).grid(row=1,column=1)
#for i in range(0,3):
#   w = Label(root, image=photo).grid(row=1,column=i)
root.mainloop()
