import cv2
#from model import FacialExpressionModel
import numpy as np
from keras.models import load_model

facec = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')
facec.load('Haarcascade/haarcascade_frontalface_default.xml')
model = load_model('expression.h5')
font = cv2.FONT_HERSHEY_SIMPLEX

def face_detection(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(img,scaleFactor=1.3,minNeighbors=5)
    
    return faces,gray_img
cam = cv2.VideoCapture(0)
    
cv2.namedWindow("test")

mood_list = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    face_detected,gray_img = face_detection(frame)
        
            
        
        #faces, faceID = ff.datapreprocessing()


    for face in face_detected:
        (x,y,w,h) = face
        roi_gray = gray_img[y:y+h,x:x+w]
        roi = cv2.resize(roi_gray, (48, 48))
        cv2.imshow("roi", roi)

        mood = model.predict(roi[np.newaxis, :, :, np.newaxis])
        name= mood_list[np.argmax(mood)]
        print(name)
cam.release()
cv2.destroyAllWindows()