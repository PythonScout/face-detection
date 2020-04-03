# face-detection
Used language Python<br>
Need the following libraries - Pillow, Numpy, OpenCV.<br>
Use pip to intall the above libraries.<br>
Create a folder and fill it with images relitive to name of folder under Images_to_train.(images accepted are .png and .jpg/.jpeg)<br>
First Run Face-train.py to train the model on basis of Images_to_train.<br>
In Face-train if any error occurs it's mainly due to the directory of "haarcascade_frontalface_alt2.xml" that is present in Cascades\data.<br>
Input the correct directory.<br>
Then use WebCam.py to Run the Webcam and test the model.
