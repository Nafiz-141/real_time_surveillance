import os
import torch
import numpy as np
from data.sphar_dataset import SPHARDataset
from models.object_detection import YOLODetector
import logging

def preprocess_dataset(data_path, output_path, sequence_length=16):
    """
    Preprocess SPHAR dataset by extracting features offline
    
    Args:
        data_path (str): Path to SPHAR dataset
        output_path (str): Path to save preprocessed features
        sequence_length (int): Length of sequence for preprocessing
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize object detector
    detector = YOLODetector(model_type='yolov8n', confidence_threshold=0.3)
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Iterate through splits
    splits = ['train', 'val', 'test']
    
    for split in splits:
        # Create dataset for current split
        dataset = SPHARDataset(data_path, sequence_length=sequence_length, split=split)
        
        # Store preprocessed data
        preprocessed_features = []
        preprocessed_labels = []
        
        logger.info(f"Preprocessing {split} split...")
        
        for video_path, label in zip(dataset.video_paths, dataset.labels):
            try:
                # Extract features offline
                features = dataset._extract_features(video_path)
                
                # Convert to tensor
                features_tensor = torch.FloatTensor(features)
                label_tensor = torch.tensor(label, dtype=torch.long)
                
                preprocessed_features.append(features_tensor)
                preprocessed_labels.append(label_tensor)
            
            except Exception as e:
                logger.error(f"Error preprocessing {video_path}: {e}")
        
        # Save preprocessed data
        split_data = {
            'features': torch.stack(preprocessed_features),
            'labels': torch.stack(preprocessed_labels)
        }
        
        torch.save(split_data, os.path.join(output_path, f'{split}_data.pt'))
        
        logger.info(f"Saved {len(preprocessed_features)} preprocessed samples for {split} split")

# Usage
if __name__ == "__main__":
    preprocess_dataset(
        data_path='./datasets/SPHAR/', 
        output_path='./preprocessed_data',
        sequence_length=16
    )