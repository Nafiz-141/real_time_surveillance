import torch
import cv2
import numpy as np
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class YOLODetector:
    """
    Wrapper class for YOLOv8 object detection
    """
    def __init__(self, model_type='yolov8n', confidence_threshold=0.5, classes=None, device=None):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_type (str): YOLOv8 model type (n, s, m, l, x)
            confidence_threshold (float): Confidence threshold for detections
            classes (list): List of class indices to detect (None for all)
            device (torch.device): Device to run inference on
        """
        self.confidence_threshold = confidence_threshold
        self.classes = classes
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLOv8 model
        try:
            self.model = YOLO(f"{model_type}.pt")
            logger.info(f"Loaded YOLOv8 model: {model_type}")
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect(self, frame):
        """
        Detect objects in a frame
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detections [x1, y1, x2, y2, confidence, class_id]
        """
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            
            # Process results
            detections = []
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    
                    # Filter by confidence and class
                    if confidence >= self.confidence_threshold:
                        if self.classes is None or class_id in self.classes:
                            detections.append([x1, y1, x2, y2, confidence, class_id])
            
            return detections
        
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            return []
    
    def annotate_frame(self, frame, detections):
        """
        Draw bounding boxes and labels on the frame
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detections [x1, y1, x2, y2, confidence, class_id]
            
        Returns:
            numpy.ndarray: Annotated frame
        """
        annotated_frame = frame.copy()
        
        for x1, y1, x2, y2, confidence, class_id in detections:
            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person {confidence:.2f}"
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y1 = max(y1, label_size[1])
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated_frame
