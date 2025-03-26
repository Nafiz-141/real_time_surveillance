import torch
from torch.utils.data import Dataset

class PreprocessedSPHARDataset(Dataset):
    def __init__(self, data_path, split='train'):
        """
        Dataset using preprocessed features
        
        Args:
            data_path (str): Path to preprocessed data
            split (str): Dataset split
        """
        preprocessed_data = torch.load(f'{data_path}/{split}_data.pt')
        self.features = preprocessed_data['features']
        self.labels = preprocessed_data['labels'].squeeze()  # Ensure 1D tensor
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_preprocessed_dataloaders(data_path, batch_size=32, num_workers=4):
    """
    Create dataloaders from preprocessed dataset
    
    Args:
        data_path (str): Path to preprocessed data
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
    Returns:
        tuple: Train, validation, and test dataloaders
    """
    train_dataset = PreprocessedSPHARDataset(data_path, split='train')
    val_dataset = PreprocessedSPHARDataset(data_path, split='val')
    test_dataset = PreprocessedSPHARDataset(data_path, split='test')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader