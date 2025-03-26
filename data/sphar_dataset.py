import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import glob
from models.object_detection import YOLODetector
import logging

logger = logging.getLogger(__name__)

class SPHARDataset(Dataset):
    """
    Dataset class for SPHAR (Surveillance Perspective Human Activity Recognition)
    """
    def __init__(self, data_path, sequence_length=20, transform=None, split='train'):
        """
        Initialize SPHAR dataset
        
        Args:
            data_path (str): Path to SPHAR dataset
            sequence_length (int): Length of sequence for LSTM
            transform (callable, optional): Transform to apply to frames
            split (str): Dataset split ('train', 'val', 'test')
        """
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.transform = transform
        self.split = split
        
        # Activity classes in SPHAR dataset
        self.activities = [
            "walking",
            "running",
            "standing",
            "sitting",
            "bending",
            "falling",
            "lying",
            "crawling",
            "fighting",
            "waving",
            "other"
        ]
        
        # Load video paths and labels
        self.video_paths, self.labels = self._load_dataset()
        
        # Initialize YOLOv8 detector
        self.detector = YOLODetector(model_type='yolov8n', confidence_threshold=0.3)
        
        logger.info(f"Loaded {len(self.video_paths)} videos for {split} split")
    
    def _load_dataset(self):
        """
        Load dataset paths and labels
        
        Returns:
            tuple: Lists of video paths and labels
        """
        video_paths = []
        labels = []
        
        # Setup train/val/test splits
        if self.split == 'train':
            split_range = (0, 0.6)
        elif self.split == 'val':
            split_range = (0.6, 0.8)
        else:  # test
            split_range = (0.8, 1.0)
        
        # Iterate through activity folders
        for activity_idx, activity in enumerate(self.activities):
            activity_dir = os.path.join(self.data_path, activity)
            if not os.path.exists(activity_dir):
                logger.warning(f"Activity directory not found: {activity_dir}")
                continue
            
            # Get all video files
            video_files = glob.glob(os.path.join(activity_dir, "*.mp4"))
            # Filter videos based on split
            start_idx = int(len(video_files) * split_range[0])
            end_idx = int(len(video_files) * split_range[1])
            split_videos = video_files[start_idx:end_idx]
            
            # Add to dataset
            for video_path in split_videos:
                video_paths.append(video_path)
                labels.append(activity_idx)
        
        return video_paths, labels
    
    def __len__(self):
        """
        Get dataset length
        
        Returns:
            int: Number of videos in dataset
        """
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        """
        Get dataset item
        
        Args:
            idx (int): Item index
            
        Returns:
            tuple: Features tensor and label
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract features from video
        features = self._extract_features(video_path)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features)
        label_tensor = torch.LongTensor([label]).squeeze()
        
        return features_tensor, label_tensor
    
    def _extract_features(self, video_path):
        """
        Extract features from video
        
        Args:
            video_path (str): Path to video file
            
        Returns:
            numpy.ndarray: Features array of shape (sequence_length, feature_size)
        """
        # Initialize features array
        # Feature size: [x1, y1, x2, y2, confidence, class_id, norm_width, norm_height, aspect_ratio]
        feature_size = 9
        features = np.zeros((self.sequence_length, feature_size))
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return features
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate frame indices to sample
            if frame_count <= self.sequence_length:
                # If video has fewer frames than sequence length, duplicate the last frame
                sample_indices = list(range(frame_count)) + [frame_count-1] * (self.sequence_length - frame_count)
            else:
                # Sample frames evenly
                sample_indices = np.linspace(0, frame_count-1, self.sequence_length, dtype=int)
            
            # Extract features from sampled frames
            for i, frame_idx in enumerate(sample_indices):
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Apply transform if available
                if self.transform:
                    frame = self.transform(frame)
                
                # Detect objects
                detections = self.detector.detect(frame)
                
                # If person detected, extract features
                if detections:
                    # Sort by confidence and get the highest confidence person detection
                    detections.sort(key=lambda x: x[4], reverse=True)
                    x1, y1, x2, y2, confidence, class_id = detections[0]
                    
                    # Calculate additional features
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 0
                    
                    # Normalize coordinates
                    norm_width = width / frame_width
                    norm_height = height / frame_height
                    
                    # Store features
                    features[i] = [x1, y1, x2, y2, confidence, class_id, norm_width, norm_height, aspect_ratio]
            
            # Release video
            cap.release()
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting features from video {video_path}: {e}")
            return features


def create_dataloaders(data_path, batch_size=32, sequence_length=20, num_workers=4):
    """
    Create dataloaders for SPHAR dataset
    
    Args:
        data_path (str): Path to SPHAR dataset
        batch_size (int): Batch size
        sequence_length (int): Length of sequence for LSTM
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: Train, validation, and test dataloaders
    """
    # Create datasets
    train_dataset = SPHARDataset(data_path, sequence_length, split='train')
    val_dataset = SPHARDataset(data_path, sequence_length, split='val')
    test_dataset = SPHARDataset(data_path, sequence_length, split='test')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader