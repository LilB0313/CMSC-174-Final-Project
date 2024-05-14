# Note, before using, please download the following libraries
# !pip install mediapipe opencv-python pandas scikit-learn
# This is the first step, to check if the camera is working which shows in real-time the frames with landmarks

# Import the necessary libraries/dependencies
import cv2
import mediapipe as mp
import csv

# Function responsible for displaying the real-time capturing of the landmarks
def realtime_display():
    # get the functions from the mediapipe for landmarking
    mp_drawing = mp.solutions.drawing_utils 
    mp_holistic = mp.solutions.holistic 

    # get access to the local camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open webcam.")
        return

    # display in real-time the mapping of the landmarks
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():          
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame.")
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)
            
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            draw_landmarks(image, results, mp_drawing, mp_holistic)
            
            capture_landmarks(results)

            # Uncomment if you wish to print the specific landmark coordinates
            # if results.right_hand_landmarks:
            #     for landmark in results.right_hand_landmarks.landmark:
            #         print(landmark)

            # if results.left_hand_landmarks:
            #     for landmark in results.left_hand_landmarks.landmark:
            #         print(landmark)

            # if results.pose_landmarks:
            #     for landmark in results.pose_landmarks.landmark:
            #         print(landmark)

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
# Function for adding the landmark/skeletal framework of the hands and body of a person
def draw_landmarks(image, results, mp_drawing, mp_holistic):
    # Specific landmarks are shown here https://i.imgur.com/AzKNp7A.png
    # Landmark for the right hand
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                            )

    # Landmark for the left hand
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                            )

    # Landmark for the whole body
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )

# Function for capturing landmarks and saving them to a CSV file
def capture_landmarks(results):
    # When running, please make both hands visible
    right_hand_landmarks = len(results.right_hand_landmarks.landmark) if results.right_hand_landmarks else 0
    left_hand_landmarks = len(results.left_hand_landmarks.landmark) if results.left_hand_landmarks else 0
    pose_landmarks = len(results.pose_landmarks.landmark) if results.pose_landmarks else 0

    # To check that the points are taken correctly, there shoud be 21 for each hand landmarks, and 33 for the pose
    # print(right_hand_landmarks, left_hand_landmarks, pose_landmarks)

    num_coords = pose_landmarks + right_hand_landmarks + left_hand_landmarks
 
    landmarks = ['class']
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        
    with open('LandMark_Coords.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)

# The main function
def main():
    realtime_display()
    # if results:  # Check if results is not None
    #     capture_landmarks(results)

if __name__ == "__main__":
    main()
