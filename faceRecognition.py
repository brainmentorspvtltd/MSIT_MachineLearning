import numpy as np
import cv2

capture = cv2.VideoCapture(0)
dataset = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

person1 = np.load('lucifer.npy').reshape(100, 50*50*3)
person2 = np.load('anmol.npy').reshape(100, 50*50*3)
data = np.concatenate( [person1, person2] )

persons = np.zeros((200,1))
persons[100:, : ] = 1
names = {0 : 'Lucifer', 1 : 'Anmol'}

def dist(x1, x2):
    return np.sqrt(sum((x2 - x1) ** 2))

def knn(x, train, k=5):
    n = train.shape[0] # 200
    distances = []
    for i in range(n):
        distances.append( dist(x, train[i]) )
    distances = np.array(distances)
    sortedIndex = np.argsort(distances)
    sortedLabels = persons[sortedIndex][:k]
    count = np.unique(sortedLabels, return_counts=True)
    return count[0][ np.argmax( count[1] ) ], distances

while True:
    ret, img = capture.read()
    if ret:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = dataset.detectMultiScale(gray, 1.3)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 5)
            myFace = img[y:y+h, x:x+w, : ]
            myFace = cv2.resize(myFace, (50,50))

            computedLabel, distances = knn(myFace.flatten(), data)
            print(distances.mean())
            if(distances.mean() > 840):
                computedLabel = "Stranger"
            else:
                computedLabel = names[computedLabel]
            
            # img, text, (x,y), font, font_size, (bgr), font_weight
            cv2.putText(img, computedLabel, (x,y), font, 1, (0,255,0), 5)
        cv2.imshow('result', img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Camera not working")
        break

capture.release()
cv2.destroyAllWindows()
