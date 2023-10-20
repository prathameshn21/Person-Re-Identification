import cv2
import dlib
import face_recognition

# Load a pre-trained face detection model (you can use other models as well)
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #Note the change

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the face recognition model (face_recognition library)
known_faces = []
known_names = []

# Initialize a dictionary to store features for each tracked individual
features_dict = {}

# Function to extract features from a face
def extract_face_features(face_image):
    face_encodings = face_recognition.face_encodings(face_image)
    if len(face_encodings) > 0:
        return face_encodings[0]
    return None

# Function to track and extract features from detected individuals
def track_and_extract_features(video_path):
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame using the face detection model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y+h, x:x+w]

            # Extract features from the detected face
            face_features = extract_face_features(face_image)

            if face_features is not None:
                # Add the features to the dictionary, you can also store the name or ID of the person
                features_dict[(x, y, x + w, y + h)] = face_features

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(10000) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'output_frame_24.jpg'  # Replace with your video file path
    track_and_extract_features(video_path)
