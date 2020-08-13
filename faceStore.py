# to capture images from webcam
import cv2
# to work on arrays
import numpy as np

# opens webcam and gives us the handle back
capture = cv2.VideoCapture(0)
# load face detection classifier
dataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# an empty to store images later on
data = []

# an infinite loop to keep the webcam on
while True:
    # starts reading data (capture image) from webcam
    ret, img = capture.read()
    # ret tells us the status of webcam -> true means webcam is on and running fine
    if ret:
        # convert image to grayscale as grayscale image is lighter (2D array, not 3D)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # detect faces in the image captured from webcam and zoom image by 30% to improve face detection
        faces = dataset.detectMultiScale(gray, 1.3)
        # put a loop and extract the details related to each face detected in the image
        # x,y -> top-left coords of face rectangle; w,h -> width and height of face rectangle
        for x,y,w,h in faces:
            # draw a rectangle on the image to show where the face was detected
            # colored_image, top-left coords of rect, bottom-right coords of rect, color of rect, border width of rect
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 5)
            # slice the face from image so we can store only the face not the complete image
            # slice rows first, then columns, then color channel (bgr)
            myFace = img[y:y+h, x:x+w, : ]
            # myFace will be a nested list of rgb values which is not going to be of same size always
            # so we need to resize all the lists for each face to a specified size and make comparisons error free
            myFace = cv2.resize(myFace, (50,50))
            # keep storing the faces in data list until you have got 100 faces
            if len(data) < 100:
                # append the face list to the data list
                data.append(myFace)
                # print length of data list to keep a check of how many faces have been captured
                print(len(data))
        # show image so that we can see where the face is detected
        cv2.imshow('result', img)
        # check for any events related to esc key or break the while loop if you have captured 100 images
        if cv2.waitKey(1) & 0xFF == 27 or len(data) == 100:
            break
    # if camera doesn't start because of some issue
    else:
        print("Camera not working")
        break

# convert list to array so that it can be stored easily and take very small storage space
data = np.array(data)
# store the array in the system
np.save('lucifer.npy', data)
# turn off the webcam
capture.release()
# close all the windows opened by cv2
cv2.destroyAllWindows()

'''
data.shape
(100, 50, 50, 3)
>>> import matplotlib.pyplot as plt
>>> data[0]
face1 = data[0]
>>> plt.imshow(face1)
<matplotlib.image.AxesImage object at 0x120aa4898>
>>> plt.show()
>>> face1 = data[50]
>>> plt.imshow(face1)
<matplotlib.image.AxesImage object at 0x120dc3748>
>>> plt.show()
'''
