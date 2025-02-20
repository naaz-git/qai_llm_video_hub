import openai
from openai import OpenAI
#from model_functions import *

client = OpenAI()  # Create a client instance

def process_command_with_gpt(command):
    """Use OpenAI's GPT to process the command and determine what action to take."""
    try:
        response = client.chat.completions.create(  # Correct method for v1.63.2
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an AI assistant for image processing."},
                {"role": "user", "content": f"Interpret this command for image processing: {command}"}
            ],
            max_tokens=50
        )
        interpreted_command = response.choices[0].message.content.strip()
        
        return interpreted_command
    except Exception as e:
        print(f"🚨 Error processing command: {e}")
        return None
'''
def process_command(interpreted_command):
    """Call the appropriate models based on the interpreted command."""
    executed_models = []  # Track executed models

    if "background removal" in interpreted_command.lower():
        executed_models.append(remove_image_background())  # Return session

    elif any(keyword in interpreted_command.lower() for keyword in ["upscale", "enhance", "increase resolution"]):
        executed_models.append(upscale_image())  # Return session

    if executed_models:
        print(f"{executed_models=}")
        return executed_models  # Return session objects
    else:
        return f"No matching model found for: {interpreted_command}"


if uploaded_file and command:
    try:
        # Process the command using GPT
        interpreted_command = process_command_with_gpt(command)
        print(f"💡 Interpreted Command: {interpreted_command}")

        if interpreted_command:
            st.write(f"💡 Interpreted Command: {interpreted_command}")

            models = process_command(interpreted_command)
            print('models[0]',models[0])
            session = models[0]
            
            # Save uploaded file
            image_path = f"temp_{uploaded_file.name}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            edited_image_path = apply_onnx_model(session, image_path)

            if edited_image_path:
                # Load and display processed image
                edited_image = cv2.imread(edited_image_path, cv2.IMREAD_UNCHANGED)
                if edited_image.shape[-1] == 4:  # RGBA
                    edited_image = cv2.cvtColor(edited_image, cv2.COLOR_BGRA2RGBA)
                elif edited_image.shape[-1] == 3:  # BGR
                    edited_image = cv2.cvtColor(edited_image, cv2.COLOR_BGR2RGB)

                # Display processed image in Streamlit
                st.image(edited_image, caption="🎨 Background Removed", use_column_width=True)

                # Provide download link for the edited image
                with open(edited_image_path, "rb") as file:
                    btn = st.download_button(
                        label="📥 Download Image",
                        data=file,
                        file_name="background_removed.png",
                        mime="image/png"
                    )

    except Exception as e:
        st.error(f"🚨 Error: {e}")'''