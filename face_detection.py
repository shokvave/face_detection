import cv2

face_cascase = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('test.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascase.detectMultiScale(gray,1.1,4)

for (x,y,w,h) in faces :
    cv2.rectangle(img, (x,y), (x +w, y+h), (225,0,0), 2)

cv2.imshow('img', img)
cv2.waitKey()

cv2.imwrite("face_detected.jpg", img)
# so we faced two problems 
# first was the incorrect naming converntion that i was following there was an extra white space "face_detection .py"
# second was the test img not being in the jpg format
# we still are running into some issues but will be fixed 
