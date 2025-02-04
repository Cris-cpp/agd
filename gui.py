import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Load Age & Gender Detection Model
model = load_model('Age_Sex_Detection.keras')

# Load OpenCV Face Detection Model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Age & Gender Detector')
top.configure(background='#CDCDCD')

# Labels for displaying results
label1 = tk.Label(top, background="#CDCDCD", font=('arial', 15, "bold"))
label2 = tk.Label(top, background="#CDCDCD", font=('arial', 15, 'bold'))
sign_image = tk.Label(top)

# Face Recognition & Cropping Function
def recognize_and_crop_face(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect Faces
   
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30))

    if len(faces) == 0:
        print("No face detected.")
        return None

    # Extract the first detected face
    x, y, w, h = faces[0]
    cropped_face = image[y:y+h, x:x+w]

    # Resize to model input size (128x128)
    cropped_face = cv2.resize(cropped_face, (128, 128))
    
    return cropped_face

# Detect Age & Gender
def Detect(file_path):
    try:
        cropped_face = recognize_and_crop_face(file_path)

        if cropped_face is None:
            label1.configure(foreground="red", text="No face detected!")
            label2.configure(text="")
            return

        # Normalize and reshape image for model
        image = np.array(cropped_face) / 255.0
        image = np.expand_dims(image, axis=0)

        # Prediction
        pred = model.predict(image)
        age = int(np.round(pred[1][0]))  # Assuming second output is age
        print(f"Raw Model Output: {pred[0][0]} (Gender), {pred[1][0]} (Age)")

        gender = "Male" if pred[0][0] > 0.5 else "Female"

        label1.configure(foreground="#011638", text=f"Age: {age}")
        label2.configure(foreground="#011638", text=f"Gender: {gender}")
      

    except Exception as e:
        print(f"Error in detection: {e}")

# Show Detection Button
def show_Detect_button(file_path):
    Detect_b = tk.Button(top, text="Detect Image", command=lambda: Detect(file_path), padx=10, pady=5)
    Detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    Detect_b.pack()

# Upload Image Function
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail((top.winfo_width()/2.25, top.winfo_height()/2.25))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        label2.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(f"Error in uploading: {e}")

# UI Elements
upload = tk.Button(top, text="Upload an Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
upload.pack(side='bottom', pady=50)

sign_image.pack(side='bottom', expand=True)
label1.pack(side="bottom", expand=True)
label2.pack(side="bottom", expand=True)

heading = tk.Label(top, text="Age and Gender Detector", pady=20, font=('arial', 20, "bold"))
heading.configure(background="#CDCDCD", foreground="#364156")
heading.pack()

top.mainloop()

