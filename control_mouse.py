# python_control_mouse.py

import pyautogui
import random

pyautogui.PAUSE = 0.5
pyautogui.FAILSAFE = True

width, height = pyautogui.size()

print("width = " + str(width))
print("height = " + str(height))

while True:
  r = random.random()
  r = r * 100
  r = r + 100
  print("r = " + str(r))
  pyautogui.moveTo(r, r, duration=0.1)
  # pyautogui.moveTo(100, 100, duration=0.25)
  # pyautogui.moveTo(200, 100, duration=0.25)
  # pyautogui.moveTo(200, 200, duration=0.25)
  # pyautogui.moveTo(100, 200, duration=0.25)