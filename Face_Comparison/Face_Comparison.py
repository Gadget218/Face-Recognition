import cv2
import face_recognition

img=cv2.imread("Messi1.webp")
rgb_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#Convert color from bgr format to rgb

#Encode the image
img_encoding = face_recognition.face_encodings(rgb_img)[0]#it couldprobably load multiple images so we use 0 as an index

img2=cv2.imread(r"images\Messi.webp")
rgb_img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

img3=cv2.imread(r"images\Elon Musk.jpg")
rgb_img3=cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img_encoding3 = face_recognition.face_encodings(rgb_img3)[0]

#Let's compare the two faces to see if they are the same
result=face_recognition.compare_faces([img_encoding],img_encoding2)
print("Result for Image 1 and 2: ",result)

result1=face_recognition.compare_faces([img_encoding],img_encoding3)
print("Result for Image 1 and 3: ",result1)

cv2.imshow("Img", img) #to see the image
cv2.imshow("Img 2", img2)
cv2.waitKey(0) #It will wait till we press a key
