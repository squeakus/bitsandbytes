import pyautogui
import time

x,y = pyautogui.size()
print("x:", x, " y:", y)

while True:
	pyautogui.moveTo(400, 400, duration=0.25)
	pyautogui.moveTo(600, 400, duration=0.25)
	pyautogui.moveTo(600, 600, duration=0.25)
	pyautogui.moveTo(400, 600, duration=0.25)
	pyautogui.click(400, 600, button='left')
	pyautogui.scroll(400)
	time.sleep(2)
	pyautogui.scroll(-400)
	time.sleep(10)