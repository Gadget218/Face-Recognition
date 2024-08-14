import os
import cv2
import pickle
import numpy as np
import face_recognition

 #Save encodings
def saveEncodings(encs, names, fname="encodings.pickle"):
    """
    Save encodings in a pickle file to be used in the future.
    """
    data = []
    d = [{"name":nm,"encoding":enc} for (nm,enc) in zip(names, encs)]
    data.extend(d)

    encodingsFile=fname

    # dump the facial encodings data to disk
    print("[INFO] serializing encodings...")
    f=open(encodingsFile, "wb")
    f.write(pickle.dumps(data))
    f.close()

#Function to read encodings
def readEncodingsPickle(fname):
    """
    Read Pickle file.
    """
    data = pickle.loads(open(fname, "rb").read())
    data=np.array(data)
    # encodings=[d["encodings"]for d in data]
    encodings = [d.get("encodings") for d in data]

    names=[d["name"] for d in data]
    return encodings,names

#function to get encodings and get face locations
def createEncodings(image):
    """
    Create face encodings for a given image and also return face locations in the given image.
    """

    #Find face locations for all faces in an image
    face_locations=face_recognition.face_locations(image)

    #Create encodings for all faces in an image
    known_encodings=face_recognition.face_encodings(image,known_face_locations=face_locations)
    return known_encodings, face_locations

    #function to compare face encodings
def compareFaceEncodings(unknown_encoding, known_encodings, known_names):
    """
    Compares face encodings to check if two faces are the same or not.
    """
    duplicateName = ""
    distance = 0.0

    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=0.5)
    best_match_index = next((index for index, match in enumerate(matches) if match), None)

    if best_match_index is not None:
        acceptBool = True
        duplicateName = known_names[best_match_index]
        distance = face_recognition.face_distance([known_encodings[best_match_index]], unknown_encoding)
    else:
        acceptBool = False

    return acceptBool, duplicateName, distance


#Save Image to new directory
def saveImageToDirectory(image,name,imageName):
    """
    Saves Images to Directory.
    """
    path=os.path.join("output", name)

    os.makedirs(path, exist_ok=True)
    cv2.imwrite(os.path.join(path, imageName),image)
#Function for creating encodings for known people
def processKnownPeopleImages(path="./People/", saveLocation="./known_encodings.pickle"):
    """
    Process images of known people and create face encodings to compare in the future. Each image should have just 1 face in it.
    """
    known_encodings = []
    known_names = []
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        # Read image
        image = cv2.imread(imgPath)
        name = img.rsplit('.')[0]
        # Resize
        image = cv2.resize(image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
        # Get locations and encodings
        encs, locs = createEncodings(image)
        for enc in encs:  # Append each encoding individually
            known_encodings.append(enc)
            known_names.append(name)  # Append the same name for each encoding from the same image

        # Show image with rectangle
        for loc in locs:
            top, right, bottom, left = loc
            cv2.rectangle(image, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
            cv2.imshow("Image", image)
            cv2.waitKey(1)
            cv2.destroyAllWindows()

    # Save all encodings after processing all images
    saveEncodings(known_encodings, known_names, saveLocation)


#Function for processing dataset Images
def processDatasetImages(path="./Dataset/",saveLocation="./dataset_encodings.pickle"):
    "Process image in dataset from where you want to separate images.It separates the images into directories of known people, groups and any unknown people images."

    people_encodings,names=readEncodingsPickle("./known_encodings.pickle")

    for img in os.listdir(path):
        imgPath=path+img

        #Read image
        image=cv2.imread(imgPath)
        orig=image.copy()

        #Resize
        image=cv2.resize(image,(0,0),fx=0.2,fy=0.2,interpolation=cv2.INTER_LINEAR)

        #Get location and encodings
        encs, locs=createEncodings(image)

        #Save image to a group image folder if more than one face is in image
        if len(locs)>1:
            saveImageToDirectory(orig,"Group",img)

        #Processing image for each face
        i=0
        knownFlag=0
        for loc in locs:
            top, right, bottom, left=loc
            unknown_encoding=encs[i]
            i+=1
            acceptBool,duplicateName,distance=compareFaceEncodings(unknown_encoding,people_encodings,names)
            if acceptBool:
                saveImageToDirectory(orig,duplicateName,img)
                knownFlag=1
            if knownFlag==1:
                print("Match Found")
            else:
                saveImageToDirectory(orig,"unknown",img)

            #Show Image
            cv2.rectangle(image,(left,top),(right,bottom), color=(255,0,0),thickness=2)
            cv2.imshow("Image",image)
            cv2.waitKey(1)
            cv2.destroyAllWindows()

def main():
    """
    Main Function.
    Returns.
    """
    datasetPath="./Dataset/"
    peoplePath="./People/"
    processKnownPeopleImages(path=peoplePath)
    processDatasetImages(path=datasetPath)
    print("Completed")

if __name__=="__main__":
    main()
