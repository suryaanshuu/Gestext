import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("sign-lang.h5")

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define a function to preprocess the camera frame
def preprocess_frame(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to match the input shape of the model (28x28)
    resized_frame = cv2.resize(gray_frame, (28, 28))
    # Normalize pixel values to range [0, 1]
    normalized_frame = resized_frame / 255.0
    # Reshape the frame to match the input shape expected by the model
    preprocessed_frame = np.expand_dims(normalized_frame, axis=-1)
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)
    return preprocessed_frame

# Function to convert class label to text representation
def class_to_text(label):
    # Define a mapping of class labels to text representations
    classes_to_text = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D',
        4: 'E',
        5: 'F',
        6: 'G',
        7: 'H',
        8: 'I',
        9: 'J',
        10: 'K',
        11: 'L',
        12: 'M',
        13: 'N',
        14: 'O',
        15: 'P',
        16: 'Q',
        17: 'R',
        18: 'S',
        19: 'T',
        20: 'U',
        21: 'V',
        22: 'W',
        23: 'X',
        24: 'Y',
        25: 'Z'
        # Add more mappings as needed
    }
    # Return the corresponding text representation for the given class label
    return classes_to_text.get(label, 'Unknown')

# Open the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera.")
else:
    detected_letters = []  # List to store detected letters
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            print("Error: Unable to read frame.")
            break

        # Define the region of interest (ROI) for sign detection
        roi = frame[100:300, 100:300]  # Example: Define a 200x200 pixel ROI at (100, 100)

        # Preprocess the ROI
        preprocessed_frame = preprocess_frame(roi)

        # Make prediction using the trained model
        predictions = model.predict(preprocessed_frame)
        predicted_class = np.argmax(predictions)
        predicted_text = class_to_text(predicted_class)

        # Display the predicted text on the frame
        cv2.putText(frame, predicted_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw a rectangle around the ROI
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Sign Language Recognition", frame)

        # Wait for user input
        key = cv2.waitKey(1) & 0xFF

        # Check if the 'SPACE' key is pressed to save the detected letter
        if key == ord(' '):
            # Add the detected letter to the list
            detected_letters.append(predicted_text)
            print("Detected letters sequence:", detected_letters)

        # Check if the 'ENTER' key is pressed to display all detected letters
        elif key == 13:  # ASCII code for 'ENTER'
            # Display the detected letters sequence
            print("Detected letters sequence:", detected_letters)
            # Clear the list for the next sequence
            detected_letters = []

        # Break the loop if 'q' key is pressed
        elif key == ord('q'):
            break

    # Release the camera and close any open windows
    cap.release()
    cv2.destroyAllWindows()