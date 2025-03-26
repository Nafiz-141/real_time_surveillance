import os
import argparse
import yaml
import logging
import cv2
import torch
from utils.logger import setup_logging
from models.object_detection import YOLODetector
from models.activity_recognition import ActivityRecognizer
from models.text_generator import TextGenerator
from interface.dashboard import Dashboard
from PyQt5.QtWidgets import QApplication
import sys

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Real-Time AI-Powered Video Surveillance System')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--source', type=str, default='0', help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--no-ui', action='store_true', help='Run without GUI')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config['logging']['level'], config['logging']['log_file'])
    logger = logging.getLogger(__name__)
    logger.info("Starting Real-Time AI-Powered Video Surveillance System")
    
    # Check GPU availability
    if config['system']['gpu_enabled'] and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU for inference")
    
    # Initialize models
    object_detector = YOLODetector(
        model_type=config['models']['object_detection']['model_type'],
        confidence_threshold=config['models']['object_detection']['confidence_threshold'],
        classes=config['models']['object_detection']['classes'],
        device=device
    )
    
    activity_recognizer = ActivityRecognizer(
        hidden_size=config['models']['activity_recognition']['hidden_size'],
        num_layers=config['models']['activity_recognition']['num_layers'],
        num_classes=config['models']['activity_recognition']['num_classes'],
        sequence_length=config['models']['activity_recognition']['sequence_length'],
        dropout=config['models']['activity_recognition']['dropout'],
        device=device
    )
    
    text_generator = TextGenerator(
        template_based=config['models']['text_generator']['template_based'],
        templates=config['models']['text_generator']['templates']
    )
    
    # Initialize video capture
    source = 0 if args.source == '0' else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return
    
    # Create frame buffer
    frame_buffer = []
    frame_count = 0
    
    if not args.no_ui:
        # Initialize GUI
        app = QApplication(sys.argv)
        dashboard = Dashboard(
            config=config,
            object_detector=object_detector,
            activity_recognizer=activity_recognizer,
            text_generator=text_generator,
            video_source=cap
        )
        dashboard.show()
        sys.exit(app.exec_())
    else:
        # Console-based processing loop
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream")
                    break
                
                frame_count += 1
                if frame_count % config['system']['frame_interval'] == 0:
                    # Detect objects
                    detections = object_detector.detect(frame)
                    
                    # Add to buffer
                    frame_buffer.append((frame, detections))
                    if len(frame_buffer) > config['system']['buffer_size']:
                        frame_buffer.pop(0)
                    
                    # Recognize activity if buffer is full
                    if len(frame_buffer) == config['system']['buffer_size']:
                        activity = activity_recognizer.recognize([d for _, d in frame_buffer])
                        
                        # Generate description
                        description = text_generator.generate(activity)
                        
                        # Display results
                        logger.info(f"Activity: {activity}, Description: {description}")
                        
                        # Display frame with annotations
                        annotated_frame = object_detector.annotate_frame(frame, detections)
                        cv2.putText(
                            annotated_frame, 
                            description, 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2
                        )
                        cv2.imshow('Real-Time Surveillance', annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
