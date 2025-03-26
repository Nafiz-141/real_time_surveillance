import torch
import torch.nn as nn
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """
    LSTM model for activity recognition
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        """
        Initialize LSTM model
        
        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden layer
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of activity classes
            dropout (float): Dropout probability
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class ActivityRecognizer:
    """
    Activity recognition using LSTM
    """
    def __init__(self, hidden_size=128, num_layers=2, num_classes=12, sequence_length=20, 
                 dropout=0.5, device=None, model_path="models/activity_recognition.pth"):
        """
        Initialize activity recognizer
        
        Args:
            hidden_size (int): Size of LSTM hidden layer
            num_layers (int): Number of LSTM layers
            num_classes (int): Number of activity classes
            sequence_length (int): Length of input sequence
            dropout (float): Dropout probability
            device (torch.device): Device to run inference on
            model_path (str): Path to pre-trained model weights
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Define input size based on bounding box features
        # [x1, y1, x2, y2, confidence, class_id, normalized_width, normalized_height, aspect_ratio]
        self.input_size = 9
        
        # Initialize LSTM model
        self.model = LSTMModel(
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"Loaded activity recognition model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load activity recognition model: {e}")
        else:
            logger.warning("No pre-trained activity recognition model provided. Using untrained model.")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Define activity classes (based on SPHAR dataset)
        self.activity_classes = [
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
    
    def preprocess_detections(self, detections_sequence):
        """
        Preprocess detection sequence for LSTM input
        
        Args:
            detections_sequence (list): List of detection lists
            
        Returns:
            torch.Tensor: Preprocessed tensor of shape (1, sequence_length, input_size)
        """
        # Initialize empty tensor
        features = np.zeros((self.sequence_length, self.input_size))
        
        # Fill tensor with available detections
        for i, detections in enumerate(detections_sequence[-self.sequence_length:]):
            idx = i % self.sequence_length  # Ensure we don't exceed sequence length
            
            if detections and len(detections) > 0:
                # Use the first person detection (highest confidence if multiple)
                detections.sort(key=lambda x: x[4], reverse=True)
                detection = detections[0]
                
                # Extract bounding box coordinates and class
                x1, y1, x2, y2, confidence, class_id = detection
                
                # Calculate additional features
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / height if height > 0 else 0
                
                # Normalize coordinates by frame dimensions (assuming 1080p)
                norm_width = width / 1920
                norm_height = height / 1080
                
                # Store features
                features[idx] = [x1, y1, x2, y2, confidence, class_id, norm_width, norm_height, aspect_ratio]
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # Add batch dimension
        
        return features_tensor
    
    def recognize(self, detections_sequence):
        """
        Recognize activity from a sequence of detections
        
        Args:
            detections_sequence (list): List of detection lists
            
        Returns:
            str: Recognized activity class
        """
        try:
            # Ensure we have enough frames
            if len(detections_sequence) < self.sequence_length:
                # Pad with empty detections if needed
                detections_sequence = [[] for _ in range(self.sequence_length - len(detections_sequence))] + detections_sequence
            
            # Preprocess detections
            features = self.preprocess_detections(detections_sequence)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(features)
                
                # Get predicted class
                _, predicted = torch.max(outputs.data, 1)
                activity_idx = predicted.item()
                
                # Get activity class name
                activity = self.activity_classes[activity_idx]
                
                return activity
        
        except Exception as e:
            logger.error(f"Error during activity recognition: {e}")
            return "unknown"

    '''
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001, save_path='models/activity_recognition.pth'):
        """
        Train the LSTM model
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            save_path (str): Path to save model weights
            
        Returns:
            dict: Training history
        """
        # Set model to training mode
        self.model.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            train_loss = 0.0
            
            # Train on batches
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            val_loss, val_accuracy = self._validate(val_loader, criterion)
            
            # Update history
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f'Model saved to {save_path}')
        
        # Set model back to evaluation mode
        self.model.eval()
        
        return history
    '''
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001, save_path='models/activity_recognition.pth'):
        """
        Train the LSTM model
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            save_path (str): Path to save model weights
            
        Returns:
            dict: Training history
        """
        # Set model to training mode
        self.model.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(num_epochs):
            train_loss = 0.0
            
            # Train on batches
            for inputs, labels in train_loader:
                # Ensure inputs and labels are on the correct device
                inputs = inputs.to(self.device)
                
                # Flatten labels and ensure they are long tensor
                labels = labels.view(-1).long().to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            val_loss, val_accuracy = self._validate(val_loader, criterion)
            
            # Update history
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        # Save model
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        logger.info(f'Model saved to {save_path}')
        
        # Set model back to evaluation mode
        self.model.eval()
        
        return history



    def _validate(self, val_loader, criterion):
        """
        Validate the model
        
        Args:
            val_loader (DataLoader): Validation data loader
            criterion (nn.Module): Loss function
            
        Returns:
            tuple: Validation loss and accuracy
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        self.model.train()
        
        return val_loss, accuracy
