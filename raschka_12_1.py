"""
Book: Machine Learning with PyTorch and Scikit-Learn
Authors: Raschka, Liu, Mirjalili

Creating a PyTorch DataLoader from existing tensors
Combining two tensors into a joint dataset (need to create a custom Dataset class)
Shuffle, batch, repeat

DataLoader returns an object of the DataLoader class, which we can use to iterate through the individual elements
of its dataset.

Combining two tensors into a joint dataset: Use case is that we have a tensor for features and a tensor for labels
and need to combine them so that we can retrieve them as tuples (features, labels) for training
"""
import torch
from torch.utils.data import DataLoader, Dataset

# create simple tensor
t = torch.arange(6, dtype=torch.float32)

#data_loader = DataLoader(t)

# for item in data_loader:
#     print(item)

# if we want to create batches from this dataset (batch size of 3:
data_loader = DataLoader(t, batch_size=3)
for i, batch in enumerate(data_loader, 1):
    print(f"Batch:{i}: {batch}, Batch size: {len(batch)}")

# Combining two tensors
# t_x holds features and t_y holds labels
torch.manual_seed(1)
t_x = torch.rand([4,3], dtype=torch.float32)
t_y = torch.arange(4)

# combine features and labels into a single tensor
class JointDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()  # calling the parent class's constructor (__init__)'
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# we create a joint dataset of t_x and t_y:
joint_dataset = JointDataset(t_x, t_y)

# create a DataLoader from the joint dataset:
# also note that when we shuffle the features and labels stay in the same order
torch.manual_seed(1)
data_loader = DataLoader(joint_dataset, batch_size=2, shuffle=True)

for i, batch in enumerate(data_loader,1):
    print(f'batch {i}: \n x: {batch[0]}, \n y: {batch[1]}')

# when training with multiple epochs, need to shuffle and iterate over the dataset by
# the desired number of epochs, Let's iterate over the batched dataset twice:
# this results in two different sets of batches.
for epoch in range(2):
    print(f'Epoch {epoch+1}')
    for i, batch in enumerate(data_loader, 1):
        print(f'batch {i}: \n x: {batch[0]}, \n y: {batch[1]}')

