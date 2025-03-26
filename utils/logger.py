import logging
import os
import json
from datetime import datetime
import sys

def setup_logging(level="INFO", log_file=None):
    """
    Setup logging configuration
    
    Args:
        level (str): Logging level
        log_file (str): Path to log file
    """
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Configure logging
    handlers = []
    
    # Add file handler if log file provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers
    )

class ActivityLogger:
    """
    Logger for detected activities
    """
    def __init__(self, log_file):
        """
        Initialize activity logger
        
        Args:
            log_file (str): Path to log file
        """
        self.log_file = log_file
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Create empty log file if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                json.dump([], f)
        
        self.logger = logging.getLogger(__name__)
    
    def log_activity(self, activity, confidence, timestamp=None):
        """
        Log activity to file
        
        Args:
            activity (str): Detected activity
            confidence (float): Confidence score
            timestamp (datetime, optional): Timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format timestamp
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp_str,
            "activity": activity,
            "confidence": confidence
        }
        
        try:
            # Read existing logs
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Add new log
            logs.append(log_entry)
            
            # Write updated logs
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            self.logger.debug(f"Logged activity: {activity}, confidence: {confidence}")
            
        except Exception as e:
            self.logger.error(f"Failed to log activity: {e}")
    
    def get_recent_activities(self, limit=10):
        """
        Get recent activities from log
        
        Args:
            limit (int): Maximum number of activities to return
            
        Returns:
            list: Recent activities
        """
        try:
            # Read logs
            with open(self.log_file, 'r') as f:
                logs = json.load(f)
            
            # Sort by timestamp (newest first)
            logs.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Limit number of logs
            recent = logs[:limit]
            
            return recent
            
        except Exception as e:
            self.logger.error(f"Failed to get recent activities: {e}")
            return []