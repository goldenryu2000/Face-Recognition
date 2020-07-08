import cv2
import numpy as np
import os


###### KNN code ###############################
def distance(v1,v2):
	#Eucledian
	return np.sqrt(((v1-v2)**2).sum())
def knn(train,test,k=5):
	dist = []
	for i in range(train.shape[0]):
		#get the vector and label
		ix = train[i,:-1]
		iy = train[i,-1]

		#complete the distance from test point
		d = distance(test,ix)
		dist.append([d,iy])

	#sort based on distance and get top K
	dk = sorted(dist,key = lambda x:x[0])[:k]
	#retrieve only the labels
	labels = np.array(dk)[: , -1]

	#get frequencies of each label
	output = np.unique(labels,return_counts = True)
	#find max frequency and corresponding label
	index = np.argmax(output[1])

	return output[0][index]
###########################################################################

#Init Camera
cap  = cv2.VideoCapture(0)
#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip = 0
dataset_path = 'Data/'
face_data = []
labels = []
class_id = 0  #labels for the given file
names = {} #mapping between id and names



#Data Preparation
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		#create a mapping between class_id and the name
		names[class_id] = fx[:-4]
		print("loaded " + fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)

		#create labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id +=1
		labels.append(target)

face_dataset = np.concatenate(face_data , axis=0)
face_labels = np.concatenate(labels,axis = 0).reshape((-1,1)) #for concatenation of train set

print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels), axis=1)
print(trainset.shape)


#Testing 

while True:
	ret,frame = cap.read()
	if ret==False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)
    
	for face in faces:
		x,y,w,h = face

	 	#Get the Region of Interest
		offset = 10
		face_section = frame[y-offset:y+offset+h , x-offset:x+offset+w]
		face_section = cv2.resize(face_section,(100,100))
	    
	    #predicted label
		out = knn(trainset,face_section.flatten())

	    #Display a name and a rectangle around it
		pred_name = names[int(out)]
		cv2.putText(frame ,pred_name , (x,y-10) , cv2.FONT_HERSHEY_SIMPLEX , 1 , (255,0,0),2, cv2.LINE_AA )
		cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,255),4)

	cv2.imshow("Faces" , frame)

	keypress = cv2.waitKey(1) & 0xFF
	if keypress == ord('q'):
		break   

cap.release()
cv2.destroyAllWindows()