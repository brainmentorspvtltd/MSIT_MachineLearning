import cv2
img = cv2.imread("img_42.jpg")
#cv2.imshow("line graph", img)
#print(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img, top left corner coords(x,y), bottom right coords(x2,y2), color (BGR), borderWidth
#cv2.rectangle(gray, (20,20), (100,100), (255,0,0), 5)
#print(gray)
#cv2.imshow("gray scale image", gray)
cv2.rectangle(img, (20,20), (100,100), (255,0,0), 5)
cv2.imshow("image", img)

# viola-jones face detection algorithm
# haar features
