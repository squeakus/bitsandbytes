"""
Simple on-screen keyboard using tkinter
Author : Ajinkya Padwad
Version 1.0
"""

import tkinter as tk
from tkinter import Entry


def main():
    """Create a keyboard and output the submission"""
    keyboard = OnscreenKeyboard()

    keyboard.keyboard.mainloop()
    print("hello!")
    print(keyboard.username)


class OnscreenKeyboard:
    """
    touch screen keyboard for handling input on raspiTFT
    """

    def __init__(self):
        self.keyboard = tk.Tk()
        self.keyboard.title("Enter User Name:")
        self.username = ""
        self.keyboard.resizable(0, 0)
        self.entry = Entry(self.keyboard, width=50)
        self.entry.grid(row=0, columnspan=15)

        self.buttons = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'BACK',
                        'p', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'CANCEL',
                        'l', 'z', 'x', 'c', 'v', 'b', 'n', 'm', '_', 'DONE']
        self.create_keyboard()

    def select(self, value):
        """
        Map buttons to alphanumeric buttons to keypress, handle delete
        cancel and submit options as well.
        """
        if value == "BACK":
            self.entry.delete(len(self.entry.get())-1, tk.END)

        elif value == "CANCEL":
            self.keyboard.destroy()
        elif value == 'DONE':
            self.username = self.entry.get()
            self.keyboard.destroy()
        else:
            self.entry.insert(tk.END, value)

    def create_keyboard(self):
        """ Add the buttons to a gridlayout 9 wide"""
        row = 1  # leave room for the text box
        col = 0

        for button in self.buttons:

            def command(x=button):
                """mapping button to function"""
                return self.select(x)

            if button == "CANCEL" or button == "DONE" or button == "BACK":
                tk.Button(self.keyboard, text=button, width=6, bg="#3c4987", fg="#ffffff",
                          activebackground="#ffffff", activeforeground="#3c4987",
                          relief='raised', padx=1, pady=1, bd=1,
                          command=command).grid(row=row, column=col)

            else:
                tk.Button(self.keyboard, text=button, width=4, bg="#3c4987", fg="#ffffff",
                          activebackground="#ffffff", activeforeground="#3c4987",
                          relief='raised', padx=1, pady=1, bd=1,
                          command=command).grid(row=row, column=col)
            col += 1

            # 9 buttons per row
            if col > 9:
                col = 0
                row += 1


if __name__ == '__main__':
    main()
