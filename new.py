import cv2
import os

# Load the face cascade classifier
cascade_path = 'haarcascade_frontalface_default.xml'
if not os.path.exists(cascade_path):
    print(f"Error: Cascade classifier file '{cascade_path}' not found.")
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)

# Load the image
img_path = 'test.jpg'
img = cv2.imread(img_path)
if img is None:
    print(f"Error: Image file '{img_path}' not found or could not be opened.")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the output
cv2.imshow('Face Detected', img)
cv2.waitKey(0)  # Wait for a key press to close the window

# Save the result
output_path = "face_detected.jpg"
cv2.imwrite(output_path, img)
print(f"Face-detected image saved as '{output_path}'.")

# Close all windows
cv2.destroyAllWindows()
