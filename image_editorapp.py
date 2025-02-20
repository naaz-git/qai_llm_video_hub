import os
import cv2
import streamlit as st
from dotenv import load_dotenv
from utils import make_dirs
from remove_bg_model import load_remove_bg_model, apply_remove_bg_model_frame  # ✅ Corrected function import
from upscale_model import load_upscale_model, apply_upscale_model_frame
from command_processor import process_command_with_gpt
import numpy as np
import tempfile

# Load environment variables
load_dotenv(verbose=True)

# Output directory
def create_out_dir():
    output_dir = 'output'
    make_dirs(output_dir)

def create_ui():
    """Create and return the Streamlit UI components."""
    st.title("🖼️ LLM-Based Image Editor")
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
    command = st.text_input("✍️ Enter your image editing command")
    return uploaded_file, command

def process_command(command):
    """Process the command to determine which model to use."""
    if any(keyword in command.lower() for keyword in ["upscale", "enhance", "increase resolution"]):
        return load_upscale_model(), "upscale"
    elif "remove background" in command.lower():
        return load_remove_bg_model(), "remove_bg"
    else:
        raise ValueError("Unsupported command")

def apply_model(session, image, model_type):
    """Apply the appropriate model based on the type."""
    if model_type == "upscale":
        return apply_upscale_model_frame(session, image)
    elif model_type == "remove_bg":
        return apply_remove_bg_model_frame(session, image)  # ✅ Updated to use correct function
    else:
        raise ValueError("Unsupported model type")

def save_processed_image(processed_image):
    """Save the processed image temporarily and return the file path."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGRA2RGBA)  # Convert BGRA to RGBA
    cv2.imwrite(temp_file.name, processed_image)
    return temp_file.name

def display_image(image_path):
    """Display the processed image and provide a download link."""
    edited_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if edited_image.shape[-1] == 4:  # RGBA
        edited_image = cv2.cvtColor(edited_image, cv2.COLOR_BGRA2RGBA)
    elif edited_image.shape[-1] == 3:  # BGR
        edited_image = cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)

    # Display processed image in Streamlit
    st.image(edited_image, caption="🎨 Edited Image", use_container_width=True)

    # Provide download link for the edited image
    with open(image_path, "rb") as file:
        st.download_button(
            label="📥 Download Image",
            data=file,
            file_name="edited_image.png",
            mime="image/png"
        )

def main():
    """Main function to handle image processing and display."""
    create_out_dir()

    # Step 1: Get UI components
    uploaded_file, command = create_ui()
    
    st_frame = st.empty()

    if uploaded_file and command:
        try:
            # Step 2: Interpret the command using GPT
            interpreted_command = process_command_with_gpt(command)
            print(f'{interpreted_command=}')

            # Step 3: Process the command and select the appropriate model
            session, model_type = process_command(command)

            # Step 4: Read the uploaded file as an image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)

            # Step 5: Apply the selected model
            processed_image = apply_model(session, image, model_type)

            # Step 6: Save the processed image temporarily
            #processed_image_path = save_processed_image(processed_image)

            # Step 7: Display the processed image and provide a download link
            st_frame.image(processed_image, use_container_width=True)

        except Exception as e:
            st.error(f"🚨 Error: {e}")

# Run the main function
if __name__ == "__main__":
    main()
