"""Get text from the user."""
import Tkinter as tk

def main():
    master = tk.Tk()
    e = tk.Entry(master)
    e.pack()
    e.focus_set()

    b = tk.Button(master, text="get", width=10, command=callback)
    b.pack()

    e = tk.Entry(master, width=50)
    e.pack()
    text = e.get()
    print "received", text
    tk.mainloop()


#    user = tk.makeentry(tk.parent, "User name:", 10)##
#    password = tk.makeentry(tk.parent, "Password:", 10, show="*")#
#    content = tk.StringVar()
#    entry = tk.Entry(tk.parent, text=tk.caption, textvariable=content)

#    text = content.get()
#    content.set(text)


def makeentry(parent, caption, width=None, **options):
    tk.Label(parent, text=caption).pack(side=tk.LEFT)
    entry = tk.Entry(parent, **options)
    if width:
        entry.config(width=width)
    entry.pack(side=tk.LEFT)
    return entry


def callback():
    print "value has been submitted"

if __name__ == '__main__':
    main()
