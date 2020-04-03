import os
import cv2
from PIL import Image       #Pillow library
import numpy as np
import pickle

#adding the FrontalFace Detection 
face_cascade= cv2.CascadeClassifier(r'haarcascade_frontalface_alt2.xml')
#main objective is to load all files and turn them into a list
#then use them to train model

#gives the base directors of the Face-train.py file
BASE_DIR =os.path.dirname(os.path.abspath(__file__))
#and use that directory to access images
image_dir=os.path.join(BASE_DIR, "images")



#empty list for training and lables
current_id=0
label_ids={}    #empty dict
x_train=[]      #pixel value
y_labels=[]     #labels id


for root, dirs, files in os.walk(image_dir):
    for file in files:      #iterate through directory and search for png and jpg files
        if file.endswith('png') or file.endswith('jpg'):
            path= os.path.join(root,file)
            label= os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
            #print(label,path)

            #labeling 
            if label in label_ids:
                pass
            else:
	    #adding items in label dict
                
                current_id+=1
                label_ids[label]=current_id

            id_=label_ids[label]
            print(label_ids)
            pil_image= Image.open(path).convert("L")  #convert(L) converts to grayscale
            image_array= np.array(pil_image, "uint8") #convert image to data arrays
            #print(image_array)

            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for(x,y,w,h) in faces:
                roi =image_array[y:y+h, x:x+w]  #region of interest or Face 
                x_train.append(roi)             #appent the data to training 
                y_labels.append(id_)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_ids,f)
    
#model training
#training model
recon=cv2.face.LBPHFaceRecognizer_create()
#training the model via data and using y_labels as np arrays
recon.train(x_train, np.array(y_labels))
#saving the trainned model
recon.save("trainner.yml")
