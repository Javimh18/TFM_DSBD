import mediapipe as mp
import numpy as np
import cv2

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

FACEMESH_LANDMARKS = 468*3 # 468 points with 3 coordinates (x, y, z) each 
POSE_LANDMARKS = 33*4 # 33 points with 4 coordinates (x, y, z and visibility) each
HAND_LANDMARKS = 21*3 # 21 points with 3 coordinates (x, y, z) each

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


def draw_styled_landmarks(image, results):

    # Draw face-mesh connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(136, 198, 255), thickness=.25, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(44, 200, 68), thickness=.25, circle_radius=.5)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=.5)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=.5)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=1), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=.5)
                             ) 

def extract_right_hand_landmarks(results):
    return np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()\
                                            if results.right_hand_landmarks\
                                            else np.zeros(HAND_LANDMARKS) 
                    
def extract_left_hand_landmarks(results):
    return np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()\
                                            if results.left_hand_landmarks\
                                            else np.zeros(HAND_LANDMARKS) 

def extract_pose_landmarks(results):
    return np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()\
                                                            if results.pose_landmarks \
                                                            else np.zeros(POSE_LANDMARKS) 

def extract_facemesh_landmarks(results):
    return np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()\
                                            if results.face_landmarks \
                                            else np.zeros(FACEMESH_LANDMARKS) 
                    
def extract_landmarks_to_np(results):
    
    right_hand_landmarks = extract_right_hand_landmarks(results)
    left_hand_landmarks = extract_left_hand_landmarks(results)
    pose_landmarks = extract_pose_landmarks(results)
    facemesh_landmarks = extract_facemesh_landmarks(results)

    return np.concatenate([right_hand_landmarks, left_hand_landmarks, pose_landmarks, facemesh_landmarks])