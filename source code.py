import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import winsound

# Mediapipe APIs
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Volume Control Library Usage 
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

# Webcam Setup
width_cam, height_cam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, width_cam)
cam.set(4, height_cam)

def play_notification_sound():
    winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)

# Mediapipe Hand Landmark Model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    max_vol_msg_time = 0
    min_vol_msg_time = 0

    while cam.isOpened():
        success, img = cam.read()

        img = cv2.flip(img, 1)  # Flip horizontally
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Finding position of Hand landmarks
        landmark_list = []
        if results.multi_hand_landmarks:
            my_hand = results.multi_hand_landmarks[0]
            for id, landmark in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                landmark_list.append([id, cx, cy])          

        # Assigning variables for Thumb and Index finger position
        if len(landmark_list) != 0:
            x1, y1 = landmark_list[4][1], landmark_list[4][2]
            x2, y2 = landmark_list[8][1], landmark_list[8][2]

            # Marking Thumb and Index finger
            cv2.circle(img, (x1, y1), 15, (255, 255, 255))  
            cv2.circle(img, (x2, y2), 15, (255, 255, 255))   
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 50:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Adjust volume based on finger distance
            vol_percentage = np.interp(length, [50, 220], [0, 100])
            vol = np.interp(length, [50, 220], [min_vol, max_vol])
            volume.SetMasterVolumeLevelScalar(vol_percentage / 100, None)

            # Draw volume bar
            vol_bar = np.interp(vol_percentage, [0, 100], [400, 150])
            cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 0), cv2.FILLED)

            # Display volume percentage
            cv2.putText(img, f'{int(vol_percentage)} %', (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Annotate volume bar
            cv2.putText(img, 'High', (90, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, 'Mid', (90, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, 'Low', (90, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Display messages for maximum and minimum volume levels
            if vol_percentage == 100:
                    cv2.putText(img, 'MAXIMUM VOLUME  REACHED', (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    play_notification_sound()
            elif vol_percentage == 0:
                    cv2.putText(img, 'MINIMUM VOLUME REACHED', (180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    play_notification_sound()
        # Adding project information at the bottom
        cv2.putText(img, "(PROJECT BY RIYA LAYEK, PARUL SRIVASTAVA, AND MADHURIMA MUKHERJEE)", (30, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Volume Control', img) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
