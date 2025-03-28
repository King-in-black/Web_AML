import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd


class AttackDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to the folder containing CSV files
            transform (callable, optional): Optional transform to be applied to a sample
        """
        self.transform = transform

        # CSV filenames and corresponding class labels
        files = ['100_forward_processed.csv', '100_backward_processed.csv',
                 '100_attack_processed.csv', '100_stop_processed.csv']
        labels_map = {
            '100_forward_processed.csv': 0,
            '100_backward_processed.csv': 1,
            '100_attack_processed.csv': 2,
            '100_stop_processed.csv': 3
        }

        samples_list = []
        labels_list = []

        # Load each CSV and assign labels to each sample (row)
        for filename in files:
            file_path = os.path.join(root_dir, filename)
            df = pd.read_csv(file_path)
            samples_list.append(df)
            labels_list.extend([labels_map[filename]] * len(df))  # one label per sample

        # Concatenate all samples into a single DataFrame
        all_df = pd.concat(samples_list, ignore_index=True)

        # Convert to NumPy array of float32 (each row should have 400 features)
        self.data = all_df.to_numpy(dtype='float32')

        # Convert labels to torch tensor
        self.labels = torch.tensor(labels_list, dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Get the idx-th sample (1D vector with 400 features)
        sample = self.data[idx]

        # Convert to tensor and reshape into [4 sensors, 20 time steps, 5 features]
        sample_tensor = torch.tensor(sample, dtype=torch.float).view(4, 20, 5)

        label = self.labels[idx]

        if self.transform:
            sample_tensor = self.transform(sample_tensor)

        return sample_tensor, label


if __name__ == '__main__':
    dataset_dir = 'dataset'
    dataset = AttackDataset(root_dir=dataset_dir)

    # Display the shape and label of the first sample (expected shape: [4, 20, 5])
    sample, label = dataset[0]
    print(f"Sample shape: {sample.shape}, Label: {label}")

    # Split the dataset into training and testing sets (e.g., 80% train / 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")

    # Create DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for batch_samples, batch_labels in train_loader:
        print(f"Batch samples shape: {batch_samples.shape}")  # [batch_size, 4, 20, 5]
        print(f"Batch labels shape: {batch_labels.shape}")  # [batch_size]
        break
