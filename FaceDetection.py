import cv2
#capture = cv2.VideoCapture(0)
#ret, img = capture.read()
#cv2.imshow("result", img)

dataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("img_42.jpg")
faces = dataset.detectMultiScale(img, 1.25)
print(faces)
for x,y,w,h in faces:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 5)
cv2.imshow("result", img)
#cv2.imwrite("result.jpg", img)
