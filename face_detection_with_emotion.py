import os
import cv2
import face_recognition
from PIL import ImageDraw,Image
from pathlib import Path

#mine

from keras.models import load_model
from time import sleep
from keras_preprocessing.image import img_to_array
from keras.preprocessing import image
import numpy as np


face_classifier = cv2.CascadeClassifier('emotion.xml')
classifier =load_model('model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
#mine

#store the names of all known person from folder knownFaces
known_faces=[]
#store the face Encoding of all known person from folder knownFaces
known_faces_encoding=[]

#It print current working directory
path=os.getcwd()
path=os.path.join(path,"knownFaces")

#it take all image file from above path
for img in os.listdir(path):
    
    #my folder contains file name thumbs.db not an image so it shows error so i neglect it
    if img!="Thumbs.db":
        img_path = os.path.join(path, img)
        image = face_recognition.load_image_file(img_path)
        image_face_encoding = face_recognition.face_encodings(image)[0]
        #stem function take only name without extension
        known_faces.append(Path(img).stem)
        known_faces_encoding.append(image_face_encoding)



# initialize the video stream
print("[INFO] starting video stream...")
#cap = cv2.VideoCapture(0)

# # loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and
    ret,frame = cap.read()
    #taking face locations and face encodings of all faces in the frame
    face_locations =face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame,face_locations)
    pil_image= Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    #for all faces in frame 
    for(top,right,bottom,left),face_encoding in zip(face_locations,face_encodings):

        #we compare face encoding of current frame with all stored encoding in known_faces_encoding list
        #compare_faces return true or false based
        #we store true , false values in matches list
        matches= face_recognition.compare_faces(known_faces_encoding,face_encoding)
        
        name = "unknown"
        #we loop over matches and if we find any true value then we take its index value
        #then takes the name of person from known_faces(name list) using above index value
        #if no true value found then name remains "unknown"
        if True in matches:
            first_match_index =matches.index(True)
            name = known_faces[first_match_index]

               
        cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)
        if name=="unknown":
              cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),4)
        else:
              cv2.rectangle(frame,(left,top),(right,bottom),(0,255,),4)
        #cv2.imshow("Frame", frame)

        #put the name using face locations in current frame
      #put a box around face detected
        #if name is unknown then box is of red color
        #else box is of green color

      # mine 
   # _, frame2 = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        #cv2.rectangle(frame2,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)



        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y+h+20)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
        
        
     #mine   

     

        

    # show the output frame
    #cv2.imshow("Frame", frame)
    #video remain active until we press 'q'
    key = cv2.waitKey(1)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
#cap.stop()
