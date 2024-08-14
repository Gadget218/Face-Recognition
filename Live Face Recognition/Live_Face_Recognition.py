
import threading

import cv2

from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) #0 sets the camera to be slected, if you have multiple cameras inser 1,2 etc to select that camera

#Let's set the height and width of camera structure(or capture):
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_Height, 480)

counter=0

face_match=False

reference_img = cv2.imread("reference.jpg") #reference image, the model will compare the capture with this image and identify whether it matches or not.

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match=True
        else:
            face_match=False
    except ValueError:
        face_match=False

while True:
    ret, frame = cap.read()

    if ret:
        if counter%30==0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),)).start() #argument is a tuple. Here we defone a thread and start it.
            except ValueError:
                pass
        counter+=1

        if face_match:
            cv2.putText(frame, "Match!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3) #displaing text and assigning its size, font, color, and thickness
        else:
            cv2.putText(frame, "No Match!", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame) #show the video of what camera is seeing

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
