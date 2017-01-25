import Tkinter
import time

class App():
    def __init__(self):
        self.root = Tkinter.Tk()
        self.label = Tkinter.Label(text="")
        self.label.pack()
        self.update_clock()
        self.root.mainloop()

    def update_clock(self):
        now = time.strftime("%H:%M:%S")
        self.label.configure(text=now)
        self.root.after(1000, self.update_clock)

app=App()
