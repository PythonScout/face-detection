import cv2
import numpy as np 
import pickle
#face model to detect face

face_cascade= cv2.CascadeClassifier(r'E:\project\Cascades\data\haarcascade_frontalface_alt2.xml')

#access to webcam use(1,2,3... for external webcams)
v=cv2.VideoCapture(0)
#bringing the model trainned data
recon=cv2.face.LBPHFaceRecognizer_create()
recon.read("trainner.yml")

label={"person_name":1}
with open("labels.pickle",'rb') as f:
    og_label=pickle.load(f)
    label={v:k for k,v in og_label.items()}  #reversing key:value to value:key

#print(label)
while(v.isOpened()):
    ret, frame = v.read()
    #converting frame to gray color
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #MODEL
    faces= face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #ITERATING THROUGH FACES
    for(x,y,w,h) in faces:
        #detecting face part
        pr_gray=gray[y:y+h, x:x+w]

        #using model to predict
        id_,conf=recon.predict(pr_gray)
        if conf>=45 and conf<=85:
            print(id_)
            print(label[id_])
            font= cv2.FONT_HERSHEY_SIMPLEX
            name=label[id_]
            color=(255,255,255)
            cv2.putText(frame,name,(x,y),font,1,color,2,cv2.LINE_AA)

        #recognise the face!!
        #(***not the perfect method***)
        
        #using image
        img_item="my-face.png"
        #saving the face part as png
        cv2.imwrite(img_item,pr_gray)
        color=(255,0,0) #BGR color (0-255) Hence its totally blue
        cv2.rectangle(frame, (x,y) ,(x+w,y+h), color, 2) #draw rectangle on face
                            #starting #ending 
                            #pt1      #pt2

    
    cv2.imshow('preview',frame)
    key=cv2.waitKey(20)
    if key==27:
        break 
    
v.release()
cv2.destroyAllWindows()
