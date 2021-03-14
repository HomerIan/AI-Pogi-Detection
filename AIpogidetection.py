import cv2
import imutils
import numpy
import os

algo = "haarcascade_frontalface_default.xml"
datasets = 'datasets'
(images, labels, names, id) = ([], [], {}, 0)
#locate datasets folder
for(subdirs, dirs, files) in os.walk(datasets):
    #folder and subfolder
    for subdir in dirs:
        #iteration to get every subfolder
        names[id] = subdir
        #creating subject path of every subfolder in folder
        subjectpath = os.path.join(datasets, subdir)
        #iteration read the files in subfolder
        for filename in os.listdir(subjectpath):
            #file path 
            path = subjectpath + '/' + filename
            label = id
            #taking each images
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
            #print(labels)
        id+=1

#size for crop image
(width, height) = (130, 100)
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

#load the algorithm
model = cv2.face.LBPHFaceRecognizer_create()
#model  = cv2.face.FisherFaceRecognizer_create()
#train the to recognize the images
model.train(images, labels)
print("training completed")

#load the algo file
haar = cv2.CascadeClassifier(algo)
cam = cv2.VideoCapture(1)
'''
address = ''
cam.open(address)
'''
cnt = 0
while True:
    _,frame = cam.read()
    frame = imutils.resize(frame, width = 600)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Obtaining Face coordinates by passing algorithm
    faces = haar.detectMultiScale(grayFrame, 1.5, 4)
    for(x,y,w,h) in faces:
        #draw rectangle on the face
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0, 255, 0), 2)
        #crop face
        face = grayFrame[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        #prediction
        #value(names, accuracy)
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255), 2)
        #check prediction < accuracy
        if prediction[1] < 90: 
            cv2.putText(frame,
                        "Pogi",
                        (x-10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (102,255,102),
                        2)
            print(names[prediction[0]])
            cnt=0
        else:
            cnt+=1
            cv2.putText(frame,
                        'Normal Face',
                        (x-10, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (102,102,255),
                        2)
    
    cv2.imshow("AI POGI-DETECTOR", frame)
    key = cv2.waitKey(10)
    #27 esc btn
    if key == 27:
        break
#cam.release()
cv2.destroyAllWindows()

