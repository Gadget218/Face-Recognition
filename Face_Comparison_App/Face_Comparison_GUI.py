import cv2
import face_recognition
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

def select_image():
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp")])
    return file_path

def compare_images():
    try:
        img_path1=select_image()
        img_path2=select_image()

        img1=cv2.imread(img_path1)
        rgb_img1=cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)#Convert color from bgr format to rgb

        #Encode the image
        img_encoding1 = face_recognition.face_encodings(rgb_img1)[0]#it couldprobably load multiple images so we use 0 as an index

        img2=cv2.imread(img_path2)
        rgb_img2=cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

        #Let's compare the two faces to see if they are the same
        result=face_recognition.compare_faces([img_encoding1],img_encoding2)
        messagebox.showinfo("Result", result)
        if result[0]: # Access the first (and only) value in the result list
            messagebox.showinfo("Result", "The faces match!")
        else:
            messagebox.showinfo("Result", "The faces do not match.")

        cv2.imshow("Img1", img1) #to see the image
        cv2.imshow("Img 2", img2)
        cv2.waitKey(0) #It will wait till we press a key
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Setup tkinter window
root = tk.Tk()
root.title("Face Comparison App")
root.geometry("300x200")

# Add button to trigger comparison
compare_button = tk.Button(root, text="Select Images to compare", command=compare_images)
compare_button.pack(pady=50)

root.mainloop()
