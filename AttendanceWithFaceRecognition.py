import cv2
import numpy as np
import face_recognition
import os
import datetime

path = 'Images'
images = []
names = []
myList = os.listdir(path)

for i in myList:
    currentImage = cv2.imread(f'{path}/{i}')
    images.append(currentImage)
    names.append(os.path.splitext(i)[0])
print(names)

def encode(images):
    encodeList = []
    for i in images:
        iConverted = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encodeI = face_recognition.face_encodings(iConverted)[0]
        encodeList.append(encodeI)
    return encodeList

encodeForKnown = encode(images)
print("Images are encoded...")

def checkAttendance(name):
    with open('list.csv','r+') as f:
        dataList = f.readlines()
        nameList = []
        for i in dataList:
            entry = i.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.datetime.now()
            datetimeString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{datetimeString}')


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgScvt = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesOnCurrentFrame = face_recognition.face_locations(imgScvt)
    encodeOnCurrentFrame = face_recognition.face_encodings(imgScvt,facesOnCurrentFrame)

    for encodedFace, faceLocation in zip(encodeOnCurrentFrame,facesOnCurrentFrame):
        match = face_recognition.compare_faces(encodeForKnown, encodedFace)
        faceDistances = face_recognition.face_distance(encodeForKnown,encodedFace)

        matchIndex= np.argmin(faceDistances)

        if match[matchIndex]:
            name = names[matchIndex].upper()
            y1,x2,y2,x1 = faceLocation
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img,(x1,y2-30),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img,name,(x1,y2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            checkAttendance(name)

    cv2.imshow('CAM',img)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()