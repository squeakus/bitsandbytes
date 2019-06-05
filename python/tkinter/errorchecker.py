import tkinter as tk
from PIL import Image, ImageTk


def main():

    changes = [{'coords': [230, 10, 290, 60], 'type':"false"},
               {'coords': [400, 100, 450, 130], 'type':"false"},
               {'coords': [100, 290, 140, 340], 'type':"false"},
               {'coords': [500, 500, 520, 520], 'type':"false"}]
    checkerwindow = ErrorChecker('cat.jpg', changes)


class ErrorChecker:
    def __init__(self, filename, changes):
        self.filename = filename
        self.changes = changes
        self.root = tk.Tk()
        self.img = ImageTk.PhotoImage(Image.open(filename))
        self.canvas = tk.Canvas(self.root, bg='black',
                                width=self.img.width(),
                                height=self.img.height())

        # listen for mouse clicks
        self.canvas.bind("<Button-1>", self.mouse_click)
        self.canvas.pack(fill="both", expand="yes")

        # load an image
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

        # draw the boxes
        for change in self.changes:
            print(change)
            draw_box(self.canvas, change['coords'], 'green3')
        self.root.mainloop()

    def mouse_click(self, event):
        print("clicked at", event.x, event.y)
        for change in self.changes:
            self.check_boundaries(event.x, event.y, change['coords'])

    def check_boundaries(self, x, y, coords):
        if x > coords[0] and x < coords[2]:
            if y > coords[1] and y < coords[3]:
                print("clicked on a box")
                draw_box(self.canvas, coords, 'blue')


def draw_box(canvas, coords, color):
    canvas.create_rectangle(coords[0], coords[1],
                            coords[2], coords[3],
                            outline=color,
                            width=2)


if __name__ == "__main__":
    main()
