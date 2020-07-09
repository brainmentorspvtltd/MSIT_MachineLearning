import cv2

capture = cv2.VideoCapture("video_1.mp4")
dataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

while True:
    ret, img = capture.read()
    if ret:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray, 1.3)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 5)
            # img, text, (x,y), font, font_size, (bgr), font_weight
            cv2.putText(img, 'Person', (x,y), font, 1, (0,255,0), 5)
        cv2.imshow('result', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Camera not working")
        break

capture.release()
cv2.destroyAllWindows()