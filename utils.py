import os
import cv2
import numpy as np

def make_dirs(output_dir):
    """Create the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
