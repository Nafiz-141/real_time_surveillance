import cv2
import numpy as np
import torch
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

def resize_frame(frame, target_size=(640, 640)):
    """
    Resize frame while maintaining aspect ratio
    
    Args:
        frame (numpy.ndarray): Input frame
        target_size (tuple): Target size (width, height)
        
    Returns:
        numpy.ndarray: Resized frame
    """
    h, w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas with target size
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center image on canvas
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Paste resized image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def normalize_detections(detections, frame_width, frame_height):
    """
    Normalize detection coordinates to [0, 1] range
    
    Args:
        detections (list): List of detections [x1, y1, x2, y2, confidence, class_id]
        frame_width (int): Frame width
        frame_height (int): Frame height
        
    Returns:
        list: Normalized detections
    """
    normalized = []
    
    for detection in detections:
        x1, y1, x2, y2, confidence, class_id = detection
        
        # Normalize coordinates
        norm_x1 = x1 / frame_width
        norm_y1 = y1 / frame_height
        norm_x2 = x2 / frame_width
        norm_y2 = y2 / frame_height
        
        normalized.append([norm_x1, norm_y1, norm_x2, norm_y2, confidence, class_id])
    
    return normalized

def extract_person_features(detections, frame_width, frame_height):
    """
    Extract features from person detections
    
    Args:
        detections (list): List of detections [x1, y1, x2, y2, confidence, class_id]
        frame_width (int): Frame width
        frame_height (int): Frame height
        
    Returns:
        numpy.ndarray: Feature vector
    """
    # Initialize feature vector
    # [center_x, center_y, width, height, aspect_ratio, area_ratio, confidence]
    features = np.zeros(7)
    
    # Get highest confidence person detection
    person_detections = [d for d in detections if d[5] == 0]  # Class 0 is person in COCO
    
    if person_detections:
        # Sort by confidence
        person_detections.sort(key=lambda x: x[4], reverse=True)
        x1, y1, x2, y2, confidence, _ = person_detections[0]
        
        # Calculate features
        center_x = (x1 + x2) / 2 / frame_width
        center_y = (y1 + y2) / 2 / frame_height
        width = (x2 - x1) / frame_width
        height = (y2 - y1) / frame_height
        aspect_ratio = width / height if height > 0 else 0
        area_ratio = (width * height)
        
        features = np.array([center_x, center_y, width, height, aspect_ratio, area_ratio, confidence])
    
    return features

def augment_frame(frame):
    """
    Apply data augmentation to frame
    
    Args:
        frame (numpy.ndarray): Input frame
        
    Returns:
        numpy.ndarray: Augmented frame
    """
    # Convert to PIL image for torchvision transforms
    frame_pil = transforms.ToPILImage()(frame)
    
    # Define augmentation transforms
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.ToPILImage()
    ])
    
    # Apply augmentation
    augmented = augmentation(frame_pil)
    
    # Convert back to numpy array
    augmented_np = np.array(augmented)
    
    return augmented_np