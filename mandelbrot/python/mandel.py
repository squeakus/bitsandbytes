from Tkinter import *

def mandel(c):
        z=0
        for h in range(0,20):
            z = z**2 + c
            if abs(z) > 2:
                break
        if abs(z) >= 2:
            return False
        else:
            return True

root = Tk()
w = Canvas(root, width=1000,height=1000)
w.pack()

for hx in range(0,1000,75):
    w.create_line(0,hx,1000,hx,fill="blue")

for hy in range(0,1000,75):
    w.create_line(hy,0,hy,1000,fill="blue")

print "Initializing..."

for x in range(0,1000):
    real = x / 500.0 - 1.5
    for y in range(0,1000):
        img = y / 400.0 - 1.5
        c = complex(real, img)
        if mandel(c):
            w.create_line(x,1000-y,x+1,1001-y,fill="black")
            w.pack()

print "Complete!"

root.mainloop()    
