import onnxruntime as ort
import cv2
import numpy as np

def load_upscale_model():
    """Load the image upscaling ONNX model."""
    ONNX_MODEL_PATH = "models/esrgan.onnx"
    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH)
        print('Image upscaling model loaded.')
        return session
    except Exception as e:
        print(f"🚨 Error loading image upscaling model: {e}")
        return None

def preprocess_frame(frame):
    """Resize and preprocess frame for upscaling model."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = cv2.resize(frame, (128, 128))  # Resize to model's expected input size
    frame = frame.astype(np.float32) / 255.0  # Normalize frame
    frame = np.transpose(frame, (2, 0, 1))  # Convert to (C, H, W)
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

def apply_upscale_model_frame(session, frame):
    """Run frame through upscaling model."""
    try:
        # Preprocess frame
        input_tensor = preprocess_frame(frame)
        inputs = {session.get_inputs()[0].name: input_tensor}
        outputs = session.run(None, inputs)

        # Process output tensor
        output_tensor = outputs[0][0]
        output_tensor = np.transpose(output_tensor, (1, 2, 0))
        output_tensor = (output_tensor * 255).clip(0, 255).astype(np.uint8)

        return cv2.cvtColor(output_tensor, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    except Exception as e:
        print(f"🚨 Error applying upscaling model: {e}")
        return frame  # Return original frame in case of error
