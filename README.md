"VOLUME CONTROL USING HAND GESTURE RECOGNITION"

This Python script allows us to control our system's volume using hand gestures captured from a webcam. It utilizes the MediaPipe library to detect hand landmarks and calculates the distance between the thumb and index finger to adjust the volume accordingly. Additionally, it provides visual feedback with a volume bar and displays messages when maximum or minimum volume levels are reached.

Features
- Control system volume using hand gestures
- Visual feedback with volume bar
- Displays messages for maximum and minimum volume levels
- Project by Riya Layek, Parul Srivastava, and Madhurima Mukherjee

Requirements
- Python 3.x
- OpenCV
- MediaPipe
- PyCaw
- Comtypes

Install the required libraries using pip:
-pip install opencv-python mediapipe pycaw comtypes

Usage
1. Run the hand_gesture_volume_control.py script.
2. Place your hand in front of the webcam.
3. Use your thumb and index finger to control the volume:
   - Bring your thumb and index finger closer to increase the volume.
   - Move your thumb and index finger apart to decrease the volume.
4. Visual feedback will be provided with a volume bar on the screen.
5. Messages will be displayed when the maximum or minimum volume levels are reached.
