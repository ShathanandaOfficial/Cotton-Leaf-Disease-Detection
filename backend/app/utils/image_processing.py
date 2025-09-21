import numpy as np
from PIL import Image
import io

def preprocess_image(image_bytes: bytes, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess image for model prediction
    
    Args:
        image_bytes: Image bytes from upload
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array
    """
    # Open image from bytes
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def validate_image_format(image_bytes: bytes) -> bool:
    """
    Validate that the uploaded file is a valid image
    
    Args:
        image_bytes: Image bytes from upload
    
    Returns:
        Boolean indicating if image is valid
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()  # Verify that it is, in fact, an image
        return True
    except (IOError, SyntaxError):
        return False