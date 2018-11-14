# eye_test.py
import pyautogui
import random
import numpy as np
import cv2

def getLeftmostEye(eyes):
  leftMostX = 9999999
  leftMostEye = (0,0,10,10)

  for (ex,ey,ew,eh) in eyes:
    print("ex = " + str(ex) + ", ey = " + str(ey) + ", ew = " + str(eh) + ", ey = " + str(eh) )
    if int(ex) < int(leftMostX):
      leftMostEye = (ex,ey,ew,eh)
      leftMostX = ex

  return leftMostEye

if __name__ == "__main__":
  print("pyautogui version = " + str(pyautogui.__version__))
  print("numpy version = " + str(np.__version__))
  print("cv2 version = " + str(cv2.__version__))

  # open cam
  cap = cv2.VideoCapture(0)

  # grab the haar casscade
  face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

  # /haarcascade_eye_tree_eyeglasses.xml
  while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Our operations on the frame come here
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = frame[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)

      for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

      lme_x,lme_y,lme_w,lme_h = getLeftmostEye(eyes)

      print("lme_x,lme_y,lme_w,lme_h = " + str((lme_x,lme_y,lme_w,lme_h)))

      leftMostEyeFrame = frame[
        y+lme_y:y+lme_y+lme_h,
        x+lme_x:x+lme_x+lme_w
      ]
      cv2.circle(frame, (lme_x+x, lme_y+y), 3, (255, 150, 150), -1)
      T = frame[70:170, 440:540]

      # cv2.cvtColor(leftMostEyeFrame, cv2.COLOR_BGR2GRAY)
      # abc = leftMostEyeFrame.astype(np.uint8)
      # abc = np.uint8(leftMostEyeFrame)
      leftMostEyeFrame_int8 = leftMostEyeFrame.astype(np.uint8)
      leftMostEyeFrame_int8 = cv2.cvtColor(leftMostEyeFrame_int8, cv2.COLOR_BGR2GRAY)
      leftMostEyeFrame_int8 = cv2.equalizeHist(leftMostEyeFrame_int8)
      # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
      # leftMostEyeFrame_int8 = clahe.apply(leftMostEyeFrame_int8)
      leftMostEyeFrame_int8 = cv2.resize(leftMostEyeFrame_int8, (0,0), fx=8, fy=8) 
      cv2.imshow("leftMostEyeFrame_int8", leftMostEyeFrame_int8)
      # cv2.equalizeHist(abc)

      # cv2.imshow('abc',abc)

    # Display Classic Frame 
    cv2.imshow('frame',frame)

    # Display Gray Frame 
    cv2.imshow('gray',gray)

    # Display the gray frame
  
    # cv2.imshow('T',T)


    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # cv2.imshow('img',img)
  # cv2.waitKey(0)
  cv2.destroyAllWindows()