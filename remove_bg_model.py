import onnxruntime as ort
import cv2
import numpy as np
import os


def load_remove_bg_model():
    """Load the background removal ONNX model."""
    ONNX_MODEL_PATH = "models/mediapipe_selfie.onnx"
    backend_path_nuget = "C:/Users/DFS/Desktop/gitrepo/nuget_packages/Microsoft.ML.OnnxRuntime.QNN.1.20.1/runtimes/win-x64/native/QnnHtp.dll"
    backend_path_qnn_sdk = "C:/Users/DFS/Desktop/qnn_sdk/qairt/2.26.0.240828/lib/aarch64-windows-msvc/QnnHtp.dll"

# sess_options = ort.SessionOptions()
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    qnn_provider_options = {
        "backend_path": backend_path_nuget
    }

    try:
        session = ort.InferenceSession(ONNX_MODEL_PATH, providers= [("QNNExecutionProvider",qnn_provider_options),"CPUExecutionProvider"],
)
        print('✅ Background removal model loaded.')
        return session
    except Exception as e:
        print(f"🚨 Error loading background removal model: {e}")
        return None

def preprocess_image_bg_model(frame):
    #Preprocess a video frame for background removal.
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, (256, 256))  # Resize to model input size

    image = image.astype(np.float32) / 255.0  # Normalize
    image = np.transpose(image, (2, 0, 1))  # Convert to (C, H, W)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def apply_remove_bg_model_frame(session, frame):
    """Run a video frame through the background removal model."""
    try:
        print('apply remove bg now')
        if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            print("🚨 Skipping frame due to invalid dimensions.")
            return frame  # Skip frame processing
        print(frame.shape[1], frame.shape[0])

        image = preprocess_image_bg_model(frame)
        print('expected', session.get_inputs()[0].shape)  # Check expected input

        # Check model input name
        input_name = session.get_inputs()[0].name
        inputs = {input_name: image}

        # Run inference
        outputs = session.run(None, inputs)
        print(f"Model Input Shape: {image.shape}")
        print(f"ONNX Output Shape: {outputs[0].shape}")

        # Extract and process mask
        mask = outputs[0][0]  # Get first output tensor
        print(f"Mask min: {mask.min()}, max: {mask.max()}")
        mask = np.clip(mask, 0, 1)  # Ensure values are within [0, 1]
        
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]  # Convert (1, H, W) → (H, W)

        # Normalize and resize mask

        mask = (mask * 255).astype(np.uint8)
     
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        # Convert frame to BGRA
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        rgba_frame[:, :, 3] = mask  # Set alpha channel

        return rgba_frame
    except Exception as e:
        print(f"🚨 Error applying background removal model: {e}")
        return frame  # Return original frame on error
