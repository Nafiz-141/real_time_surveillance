import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QPushButton, QTextEdit, QSplitter)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer
from utils.visualization import create_activity_histogram, create_activity_timeline
from utils.logger import ActivityLogger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from datetime import datetime, timedelta

class Dashboard(QMainWindow):
    def __init__(self, config, object_detector, activity_recognizer, text_generator, video_source):
        super().__init__()
        
        # Store models and configuration
        self.config = config
        self.object_detector = object_detector
        self.activity_recognizer = activity_recognizer
        self.text_generator = text_generator
        self.video_source = video_source
        
        # Initialize activity logger
        self.activity_logger = ActivityLogger(config['logging']['activity_log'])
        
        # Track last logged activity
        self.last_logged_activity = None
        self.last_log_time = datetime.min
        self.log_interval = timedelta(seconds=3)  # 3-second interval between logs
        
        # Setup main window
        self.setWindowTitle("Real-Time Surveillance Dashboard")
        self.resize(config['interface']['window_width'], 
                    config['interface']['window_height'])
        
        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Create video and info splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Video display area
        self.video_label = QLabel("Video Stream")
        self.video_label.setAlignment(Qt.AlignCenter)
        
        # Information panel
        info_panel = QWidget()
        info_layout = QVBoxLayout()
        
        # Activity log
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        
        # Activity charts
        self.activity_histogram = FigureCanvas(plt.Figure(figsize=(5, 4)))
        self.activity_timeline = FigureCanvas(plt.Figure(figsize=(5, 4)))
        
        info_layout.addWidget(QLabel("Activity Log"))
        info_layout.addWidget(self.activity_log)
        info_layout.addWidget(QLabel("Activity Distribution"))
        info_layout.addWidget(self.activity_histogram)
        info_layout.addWidget(QLabel("Activity Timeline"))
        info_layout.addWidget(self.activity_timeline)
        
        info_panel.setLayout(info_layout)
        
        # Add to splitter
        splitter.addWidget(self.video_label)
        splitter.addWidget(info_panel)
        splitter.setSizes([700, 300])  # Default split
        
        main_layout.addWidget(splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Frame buffer and processing
        self.frame_buffer = []
        
        # Setup timer for video processing
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(self.config['interface']['update_interval'])
    
    def process_frame(self):
        # Read frame from video source
        ret, frame = self.video_source.read()
        if not ret:
            return
        
        # Detect objects
        detections = self.object_detector.detect(frame)
        
        # Update frame buffer
        self.frame_buffer.append(detections)
        if len(self.frame_buffer) > self.config['system']['buffer_size']:
            self.frame_buffer.pop(0)
        
        # Recognize activity when buffer is full
        if len(self.frame_buffer) == self.config['system']['buffer_size']:
            activity = self.activity_recognizer.recognize(self.frame_buffer)
            description = self.text_generator.generate(activity)
            
            # Rate-limited logging
            current_time = datetime.now()
            should_log = (
                self.last_logged_activity != activity or  # Different activity
                current_time - self.last_log_time >= self.log_interval  # Enough time passed
            )
            
            if should_log:
                # Log activity
                self.activity_logger.log_activity(activity, 0.75)
                
                # Update tracking variables
                self.last_logged_activity = activity
                self.last_log_time = current_time
                
                # Update activity log
                self.update_activity_log(description)
                
                # Update visualizations
                self.update_activity_charts()
        
        # Annotate and display frame
        annotated_frame = self.object_detector.annotate_frame(frame, detections)
        self.display_frame(annotated_frame)
    
    def display_frame(self, frame):
        # Convert frame to QImage
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image to fit label
        scaled_image = qt_image.scaled(
            self.video_label.width(), 
            self.video_label.height(), 
            Qt.KeepAspectRatio
        )
        
        # Display image
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))
    
    def update_activity_log(self, description):
        # Add new description to log
        current_log = self.activity_log.toPlainText()
        updated_log = f"{description}\n{current_log}"
        self.activity_log.setPlainText(updated_log)
    
    def update_activity_charts(self):
        # Get recent activities
        activities = self.activity_logger.get_recent_activities(limit=50)
        
        # Update histogram
        hist_fig = create_activity_histogram(activities)
        self.activity_histogram.figure = hist_fig
        self.activity_histogram.draw()
        
        # Update timeline
        timeline_fig = create_activity_timeline(activities)
        self.activity_timeline.figure = timeline_fig
        self.activity_timeline.draw()