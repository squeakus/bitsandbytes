from Tkinter import*

class GUI: 

      def __init__(self, master): 

          self.master = master
          frame = Frame(self.master)
          frame.pack()

          self.label = Label(frame, {"text": "This is just a test program"})
          self.label.pack()

          self.button = Button(frame)
          self.button.configure(text = "OK", background = "blue", 
                foreground = "white", command = self.buttonClick); 
          self.button.pack(side = LEFT)


      def buttonClick(self): 

          self.button1 = Button(self.master, text = "Click")
          self.button1.pack(side = LEFT)




if __name__ == "__main__": 
   root = Tk()
   root_gui = GUI(root)
   root.mainloop()
