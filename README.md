To use this facial recognition code, please follow these steps:

    Ensure that you have the required dependencies installed, including OpenCV, NumPy, and TensorFlow.
    Download the Haar cascade file for face detection and save it as "haarcascade_frontalface_default.xml" in the same directory as this code.
    Download the pre-trained TensorFlow model for object recognition and save it as "model.h5" in the same directory as this code.
    Prepare a reference image that you want to compare the detected faces with and save it as "test.PNG" in the same directory.
    Connect a webcam to your computer and ensure it is properly set up.
    Run the code and grant necessary permissions for accessing the webcam.
    The code will continuously capture frames from the webcam, detect and label faces, and display the results in a window.
    If a detected face matches the reference image above a certain threshold, an alert message will be displayed on the frame.
    Press the 'q' key to exit the program.
hii stuff inatumia web cam juu hyo ndio nlikua natumia kutest.so ukitaka kuweka cctv utachange kiasi tu from "cap = cv2.VideoCapture(0)" to 
"cap = cv2.VideoCapture("rtsp://username:password@your_cctv_camera_url")"..this utakua unacapture image from CCTV.
also this code compares the captured image with one photo,you will need to connect it to a database with images, example employees images.for the code to compare the captured image from cctv to a data base you might also need some few changes here and there,,worry not!!,,i gatchu!!



import cv2
import numpy as np
import mysql.connector

# Load the cascade classifier for face detection
face_cascade = cv2.CascadeClassifier("C:/Users/Admin/PycharmProjects/haarcascade_frontalface_default.xml")

# Load the TensorFlow model for object recognition
model = keras.models.load_model("model.h5")

# Establish a connection to MySQL database
conn = mysql.connector.connect(
    host="your_host",
    user="your_username",
    password="your_password",
    database="your_database"
)

# Create a cursor to interact with the database
cursor = conn.cursor()

# Retrieve the image from the database
query = "SELECT image FROM images WHERE image_id = %s"
image_id = 1  # Adjust this based on your image ID in the database
cursor.execute(query, (image_id,))
result = cursor.fetchone()
image_blob = result[0]

# Convert the image blob to numpy array
nparr = np.frombuffer(image_blob, np.uint8)

# Decode the numpy array to image
ref_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Rest of the code remains the same as before
    # ...

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

# Close the database connection
cursor.close()
conn.close()


hapa Tutatumia:
mysql.connector library to establish a connection to a MySQL database. Make sure to replace the placeholders 'your_host', 'your_username', 'your_password', and 'your_database' with the actual connection details of your MySQL database.

Assuming you have a table named images in your database, the code retrieves the image blob from the database using a query. Modify the query to suit your database schema and table structure. The image ID is specified as image_id = 1 in this example, so adjust it based on the ID of the image you want to retrieve from the database.

The retrieved image blob is then converted to a NumPy array using np.frombuffer, and subsequently, the NumPy array is decoded to an OpenCV image using cv2.imdecode.

After that, you can continue with the rest of the code as before to perform face detection, object recognition, and image matching using the retrieved image.

Usisahau kuhandle errors and exceptions appropriately when working with databases to ensure proper connection management and data retrieval.usiteseke bure..hahaha!!..good luck!!
