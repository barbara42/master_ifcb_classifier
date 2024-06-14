# dataloader.py: 
# Functions to load data into PyTorch DataLoaders, with support
# for train/val/test splits, and class balancing.

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import os
from PIL import Image
import json
import torch

class CustomDataset(Dataset):
    """
    A custom dataset class that loads images based on a CSV file containing image ids, labels,
    and data splits (train, val, test).
    """
    def __init__(self, data_dir, csv_file, split, transform=None):
        """
        Initializes the dataset.

        Args:
            data_dir (str): Path to the directory containing images.
            csv_file (str): Path to the CSV file containing image ids, labels, and splits.
            split (str): The split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        csv_file = pd.read_csv(csv_file)

        # Count the occurrences of each class
        label_counts = csv_file['label'].value_counts()

        # Filter out classes with less than min_data_points
        min_data_points = 3
        self.filtered_labels = label_counts[label_counts >= min_data_points].index.tolist()
        self.df = csv_file[csv_file['label'].isin(self.filtered_labels)]
        
        self.labels = self.df['label'].unique()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Store ignored classes information
        self.ignored_classes = label_counts[label_counts < min_data_points].index.tolist()

        self.split = split
        self.transform = transform
        self.data = self.df[self.df['split'] == split]

    def __len__(self):
        """
        Returns the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an item at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: (image, label) where image is the transformed image and label is the class label.
        """
        img_name = os.path.join(self.data_dir, self.data.iloc[idx]['image_name'])
        image = Image.open(img_name).convert('RGB')
        #label = self.data.iloc[idx, 1]
        label = self.label_to_idx[self.data.iloc[idx]['label']]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_balanced_sampler(dataset):
    """
    Creates a sampler for balancing classes in a dataset.

    Args:
        dataset (Dataset): The dataset for which to create a sampler.

    Returns:
        WeightedRandomSampler: A sampler that can balance classes.
    """
    class_counts = dataset.data['label'].value_counts().to_dict()
    class_weights = {class_id: 1.0/count for class_id, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in dataset.data['label']]

    return WeightedRandomSampler(sample_weights, len(sample_weights))

def get_dataloaders(data_dir, label_path, batch_size=32, num_workers=1):
    """
    Prepares DataLoader instances for train, validation, and test sets.

    Args:
        data_dir (str): Path to the directory with images.
        label_path (str): Path to the CSV file with labels and splits.
        batch_size (int, optional): Batch size. Defaults to 32.
        num_workers (int, optional): Number of worker processes. Defaults to 4.

    Returns:
        dict: A dictionary with DataLoader instances for 'train', 'val', and 'test' splits.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    datasets = {split: CustomDataset(data_dir, label_path, split, transform) for split in ['train', 'val', 'test']}
    samplers = {split: create_balanced_sampler(datasets[split]) for split in ['train', 'val', 'test']}

    dataloaders = {
        split: DataLoader(datasets[split], batch_size=batch_size, sampler=samplers[split], num_workers=num_workers)
        for split in ['train', 'val', 'test']
    }

    return dataloaders

# Note: The actual function calls and usage would be outside this script, typically in the train.py, fine_tune.py, and predict.py scripts.
# This implementation assumes the CSV file has columns ['image_id', 'label', 'split'] where 'split' is one of ['train', 'val', 'test'].

def save_label_to_idx(label_to_idx_dict, output_dir, filename="label_to_idx.json"):
    """
    Saves the label to index mapping to a JSON file within the specified output directory.

    Args:
        label_to_idx_dict (dict): A dictionary mapping labels to indices.
        output_dir (str): The directory where the file should be saved.
        filename (str): The name of the file to save the mapping to. Defaults to "label_to_idx.json".
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Define the full path for the output file
    file_path = os.path.join(output_dir, filename)
    
    # Save the dictionary to a JSON file
    with open(file_path, 'w') as f:
        json.dump(label_to_idx_dict, f, indent=4)
    
    print(f"Label to index mapping saved to {file_path}")