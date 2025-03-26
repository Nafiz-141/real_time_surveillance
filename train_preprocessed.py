import torch
import os
import logging
# Import the new preprocessed dataset loader
from data.preprocessed_dataset import create_preprocessed_dataloaders
from models.activity_recognition import ActivityRecognizer

def train_activity_recognition(config_path='config/config.yaml'):
    """
    Train the LSTM activity recognition model using preprocessed data
    """
    # Load configuration
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() and config['system']['gpu_enabled'] else 'cpu')
    
    # Dataset configuration
    preprocessed_data_path = './preprocessed_data'
    model_config = config['models']['activity_recognition']
    
    # Create dataloaders from preprocessed data
    train_loader, val_loader, test_loader = create_preprocessed_dataloaders(
        data_path=preprocessed_data_path,
        batch_size=32
    )
    
    # Initialize activity recognizer
    recognizer = ActivityRecognizer(
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        num_classes=model_config['num_classes'],
        sequence_length=10,  # Match preprocessed sequence length
        dropout=model_config['dropout'],
        device=device
    )
    
    # Training parameters
    num_epochs = 50
    learning_rate = 0.001
    model_save_path = 'models/activity_recognition.pth'
    
    # Train the model
    history = recognizer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_path=model_save_path
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_accuracy = recognizer._validate(test_loader, torch.nn.CrossEntropyLoss())
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
    
    return history

if __name__ == "__main__":
    train_activity_recognition()
