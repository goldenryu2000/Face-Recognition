import cv2
import numpy as np
cap  = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
face_data = []
skip = 0
file_name = input("Enter Name: ")
dataset_path = './Data/'
while True:
    ret,frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret==False:
        continue
    faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
    faces = sorted(faces, key= lambda f:f[2]*f[3], reverse=True)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame , (x,y) , (x+w,y+h) , (255,0,0) , 4)
        offset = 10
        face_section = frame[y-offset:y+offset+h , x-offset:x+offset+w]
        face_section = cv2.resize(face_section,(100,100))
        skip+=1
        if skip%10==0:
            face_data.append(face_section)
            print(len(face_data))           
    cv2.imshow("frame" , frame)
    cv2.imshow("Face" , face_section)
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord('q'):
        break        
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
np.save(dataset_path+file_name+'.npy',face_data)
print("data Saved Succesfully") 
cap.release()
cv2.destroyAllWindows()

