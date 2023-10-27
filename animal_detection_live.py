import cv2
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load pre-trained ResNet-50 model
model = ResNet50(weights='imagenet')

# Function to recognize animal and draw bounding box
def recognize_animal_and_draw_box(frame):
    # Preprocess the frame for model prediction
    frame1 = cv2.resize(frame, (224, 224))
    x = np.expand_dims(frame1, axis=0)
    x = preprocess_input(x)

    # Make predictions using the pre-trained ResNet-50 model
    predictions = model.predict(x)

    # Decode predictions to human-readable labels
    animal_predictions = decode_predictions(predictions, top=1)[0]
    predicted_animal = animal_predictions[0][1]

    # Draw bounding box and label on the frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 0)  # Green color for the bounding box
    cv2.putText(frame, predicted_animal, (10, 30), font, 1, color, 2, cv2.LINE_AA)

    return frame

# Main function for real-time animal recognition
def animal_recognition():
    # Open a video capture stream (0 for the default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the video stream
        ret, frame = cap.read()

        # Recognize animal and draw bounding box
        frame_with_box = recognize_animal_and_draw_box(frame)

        # Display the frame with bounding box and label
        cv2.imshow('Animal Recognition', frame_with_box)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the main animal recognition function
animal_recognition()
