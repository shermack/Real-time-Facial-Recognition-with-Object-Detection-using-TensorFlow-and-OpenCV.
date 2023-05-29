import cv2
import numpy as np
from tensorflow import keras

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("C:/Users/Admin/PycharmProjects/haarcascade_frontalface_default.xml")

# Load the TensorFlow model for object recognition
model = keras.models.load_model("model.h5")

# Load the reference image for object recognition
ref_img = cv2.imread("test.PNG")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Crop the face from the frame
        face = frame[y:y+h, x:x+w]

        # Resize the face to the input size of the model
        face = cv2.resize(face, (224, 224))

        # Normalize the face pixels
        face = (face / 255.0).astype(np.float32)

        # Expand the dimensions of the face from (224, 224, 3) to (1, 224, 224, 3)
        face = np.expand_dims(face, axis=0)

        # Predict the objects in the face
        predictions = model.predict(face)

        # Get the class with the highest confidence
        class_idx = np.argmax(predictions[0])

        # Get the class name from the class index
        class_names = ['class1', 'class2', 'class3']
        class_name = class_names[class_idx]

        # Put the class name on the frame
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Compare the face with the reference image
        res = cv2.matchTemplate(face, ref_img, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        if np.max(res) > threshold:
            # Display an alert message

            cv2.putText(frame, "MATCH FOUND!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()