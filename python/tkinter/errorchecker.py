import tkinter as tk
from PIL import Image, ImageTk


def main():
    filename = 'cat.jpg'
    root = tk.Tk()
    img = ImageTk.PhotoImage(Image.open(filename))
    canvas = tk.Canvas(root, bg='black',
    				   width=img.width(), 
    				   height=img.height())
    canvas.pack(fill="both", expand="yes")

    # load an image
    canvas.create_image(0, 0, image=img, anchor=tk.NW)

    # draw the boxes
    canvas.create_rectangle(230, 10, 290, 60,
                            outline="#f11", width=2)
    root.mainloop()


if __name__ == "__main__":
    main()
