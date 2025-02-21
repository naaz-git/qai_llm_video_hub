import os
import cv2
import streamlit as st
import tempfile
from dotenv import load_dotenv
from utils import make_dirs
from remove_bg_model import load_remove_bg_model, apply_remove_bg_model_frame
from upscale_model import load_upscale_model, apply_upscale_model_frame
from command_processor import process_command_with_gpt

# Load environment variables
load_dotenv(verbose=True)

# Output directory
def create_out_dir():
    output_dir = 'output'
    make_dirs(output_dir)

def create_ui():
    """Create and return the Streamlit UI components."""
    st.title("📹 LLM-Based Video Editor")

    # Upload video first before asking for command
    uploaded_file = st.file_uploader("📂 Upload a video file", type=["mp4", "avi", "mov"])
    
    # If a file is uploaded, save it to a temporary location
    video_source = None
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_source = temp_video.name  # Store file path

    # If no file is uploaded, give user option for webcam
    if video_source is None:
        use_webcam = st.checkbox("🎥 Use Webcam Instead")
        if use_webcam:
            video_source = 0  # Webcam source

    # Now take the command after video is selected
    command = st.text_input("✍️ Enter your video editing command")
    start_processing = st.button("🎥 Start Processing Video", disabled=(video_source is None))

    return command, start_processing, video_source

def process_command(command):
    """Process the command to determine which model to use."""
    if any(keyword in command.lower() for keyword in ["upscale", "enhance", "increase resolution"]):
        return load_upscale_model(), "upscale"
    elif "remove background" in command.lower():
        return load_remove_bg_model(), "remove_bg"
    else:
        raise ValueError("Unsupported command")

def apply_model(session, frame, model_type):
    """Apply the selected model to a single video frame."""
    if model_type == "upscale":
        return apply_upscale_model_frame(session, frame)
    elif model_type == "remove_bg":
        return apply_remove_bg_model_frame(session, frame)
    else:
        raise ValueError("Unsupported model type")

def start_video_processing(command, video_source):
    """Process the selected video source (uploaded file or webcam)."""
    try:
        if video_source is None:
            st.error("🚨 No video source selected.")
            return
        
        # Interpret command using GPT
        interpreted_command = process_command_with_gpt(command)
        print(f'{interpreted_command=}')
        
        # Load the selected model
        session, model_type = process_command(command)
        
        cap = cv2.VideoCapture(video_source)
        st_frame = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video. Check your webcam or file.")
                break
            
            # Process frame using selected model
            processed_frame = apply_model(session, frame, model_type)

            # Convert BGR to RGB for Streamlit display
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGRA2RGBA)

            # Display the processed frame
            st_frame.image(processed_frame, use_container_width=True)

        cap.release()
    
    except Exception as e:
        st.error(f"🚨 Error: {e}")

def main():
    """Main function to handle video processing."""
    create_out_dir()
    
    # Step 1: Get UI components
    command, start_processing, video_source = create_ui()

    if start_processing and command and video_source is not None:
        start_video_processing(command, video_source)

if __name__ == "__main__":
    main()
