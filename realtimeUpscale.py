import streamlit as st
import cv2
import numpy as np

st.title("Live Camera Processing with OpenCV - Face Detection and Upscaling")

# Initialize session state for controlling camera
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# Function to start camera
def start_camera():
    st.session_state.camera_running = True

# Function to stop camera
def stop_camera():
    st.session_state.camera_running = False

# Start and Stop Buttons
st.button("Start Camera", on_click=start_camera)
st.button("Stop Camera", on_click=stop_camera)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# OpenCV Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Placeholders for displaying video frames
frame_placeholder_original = st.empty()
frame_placeholder_modified = st.empty()

# Upscaling factor
scale_factor = 2  # Change this value for higher or lower upscaling

# Camera processing loop
while st.session_state.camera_running:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture image")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame (only for the original frame)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around faces in the original frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Upscale the frame (resize) without any faces drawn
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    upscaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Adjust the detected face coordinates for the upscaled frame
    for (x, y, w, h) in faces:
        # Scale the face coordinates to match the upscaled frame size
        x_scaled = int(x * scale_factor)
        y_scaled = int(y * scale_factor)
        w_scaled = int(w * scale_factor)
        h_scaled = int(h * scale_factor)
        
        # Draw rectangles around faces in the upscaled frame
        cv2.rectangle(upscaled_frame, (x_scaled, y_scaled), (x_scaled + w_scaled, y_scaled + h_scaled), (0, 255, 0), 2)

    # Convert both original and upscaled frames to RGB for Streamlit display
    original_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    upscaled_frame_rgb = cv2.cvtColor(upscaled_frame, cv2.COLOR_BGR2RGB)

    # Update the original feed window with the original frame (display in smaller size)
    frame_placeholder_original.image(original_frame_rgb, channels="RGB", use_container_width=False, width=320)  # Smaller size

    # Update the modified feed window with the upscaled frame (display in larger size)
    frame_placeholder_modified.image(upscaled_frame_rgb, channels="RGB", use_container_width=False, width=640)  # Larger size

# Release the camera when stopped
cap.release()
cv2.destroyAllWindows()
