import keyboard
from win10toast import ToastNotifier


def main():
    keylogger = Keylogger(interval=10)
    keylogger.start()


class Keylogger:
    def __init__(self, interval):
        self.count = 0
        self.interval = interval
        self.toaster = ToastNotifier()

    def callback(self, event):
        print(event.name)
        self.count += 1
        if self.count % self.interval == 0:
            self.toaster.show_toast("Warning", f"You have pressed {self.count} keys")

    def start(self):
        keyboard.on_release(callback=self.callback)
        keyboard.wait("esc")


if __name__ == "__main__":
    main()
