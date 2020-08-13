import numpy as np
import cv2

capture = cv2.VideoCapture(0)
dataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# a custom font to use while printing text on image
font = cv2.FONT_HERSHEY_COMPLEX

# load images and flatten each image
person1 = np.load('lucifer.npy').reshape(100, 50*50*3)
# load images of the other person also and do the flattening
person2 = np.load('anmol.npy').reshape(100, 50*50*3)
# conactenate images of both persons to form a dataset
data = np.concatenate( [person1, person2] )

# create an array of size (200) having 0s
persons = np.zeros(200) # if error comes use shape (200,1)
# convert half the values at the end to 1 to depict person2 images
persons[100:] = 1
# give names to persons, 0 will depict Lucifer, 1 will depict Anmol
names = {0 : 'Lucifer', 1 : 'Anmol'}

# define a function to calculate distance between two images
# distance means differences between each pixel of two images
# x1 is image from webcam, x2 is some image from datastore
# it will be executed 15,00,000 times
def dist(x1, x2):
    # count distance between each pixel (using distance formula) and add all the values together (7500 values)
    return np.sqrt(sum((x2 - x1) ** 2))

# define a fn to apply knn algo
# x is image from webcam, train is complete dataset of images
# k=5 means select nearest 5 neighbours while applying knn
# k=5 means select the 5 most similar images (to the webcam images) to do prediction
def knn(x, train, k=5):
    # shape of train is (200,7500)
    n = train.shape[0] # 200 images
    # an empty list to store differences between webcam image and 200 dataset images
    distances = []
    # run the loop for as many times as you have dataset images
    for i in range(n):
        # calculate distance between webcam image and a particular datastore image
        # and append the distance in distances list
        distances.append( dist(x, train[i]) )
    # convert list into array as we want to apply array-specific methods
    distances = np.array(distances)
    # sort out the indexes based on the values of distances
    sortedIndex = np.argsort(distances)
    # sort persons array based on sortedIndex and then slice out only k values
    sortedLabels = persons[sortedIndex][:k]
    # count no of times label 0 and 1 are repeating in the sortedLabels array
    count = np.unique(sortedLabels, return_counts=True)
    # count -> (labels_array, count_array)
    # find the index of maximum number in count[1] array
    # return the computed labels and distances array also
    return np.argmax( count[1] ), distances
    #return count[0][ np.argmax( count[1] ) ], distances

while True:
    ret, img = capture.read()
    if ret:
        # resize(image, desired size of image (by default it is actual size), scale factor for x-axis, scale factor for y-axis)
        # scale image horizontally and vertically
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray, 1.3)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 5)
            myFace = img[y:y+h, x:x+w, : ]
            myFace = cv2.resize(myFace, (50,50))
            # we need to flatten our webcam image as all datastore images are also 1D 
            computedLabel, distances = knn(myFace.flatten(), data)
            print(distances.mean())
            # find mean, if mean is greater than 840 then knn was not able to recognise the person in webcam image
            if(distances.mean() > 840):
                computedLabel = "Stranger"
            else:
                # if person was recognised, get the name of that person from names dictionary
                computedLabel = names[computedLabel]
            
            # img, text, (x,y), font, font_size, (bgr), font_weight-1 t0 9
            cv2.putText(img, computedLabel, (x,y), font, 1, (0,255,0), 5)
        cv2.imshow('result', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Camera not working")
        break

capture.release()
cv2.destroyAllWindows()

'''
distances = [12, 9, 8, 34, 12, 3, 2, 5]
>>> distances.sort()
>>> distances
[2, 3, 5, 8, 9, 12, 12, 34]
>>> distances[0:2]
[2, 3]
>>> distances[0:3]
[2, 3, 5]
>>> distances = [12, 9, 8, 34, 12, 3, 2, 5]
>>> import numpy as np
>>> distances = np.array(distances)
>>> sortedIndex = np.argsort(distances)
>>> sortedIndex
array([6, 5, 7, 2, 1, 0, 4, 3])
>>> persons = [0,0,0,0,1,1,1,1]
>>> persons[sortedIndex]
Traceback (most recent call last):
  File "<pyshell#11>", line 1, in <module>
    persons[sortedIndex]
TypeError: only integer scalar arrays can be converted to a scalar index
>>> persons = np.array([0,0,0,0,1,1,1,1])
>>> persons[sortedIndex]
array([1, 1, 1, 0, 0, 0, 1, 0])
>>> persons[sortedIndex][:3]
array([1, 1, 1])
>>> sortedLabels = persons[sortedIndex][:3]
>>> np.unique(sortedLabels)
array([1])
>>> sortedLabels = np.array([1,1,1,0,0])
>>> np.unique(sortedLabels)
array([0, 1])
>>> np.unique(sortedLabels, return_counts = True)
(array([0, 1]), array([2, 3]))
>>> count = np.unique(sortedLabels, return_counts = True)
>>> count[0]
array([0, 1])
>>> count[1]
array([2, 3])
>>> np.argmax(count[1])
1
>>> count[0][np.argmax(count[1])]
'''
