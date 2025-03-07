import torch
import random
from collections import defaultdict, Counter


def get_HCP_Age(path):
        """
        Load and format the two dataframes as a list of snapshots
        """
        data1 = torch.load(path)
        
        labels = data1['labels']      
        batches = data1['batches']    

        
        label_counts = Counter(labels)
        print("Label counts:", label_counts)

        
        indices_by_class = defaultdict(list)
        for idx, label in enumerate(labels):
            indices_by_class[label].append(idx)

        # Choose a split ratio
        train_ratio = 0.8
        train_indices = []
        test_indices = []

        #shuffle indices and split
        for cls in [0, 1, 2]:
            indices = indices_by_class[cls]
            random.shuffle(indices)
            split_point = int(len(indices) * train_ratio)
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])

        train_dataset = [batches[i] for i in train_indices]
        test_dataset = [batches[i] for i in test_indices]

        # Shuffle the datasets
        random.shuffle(train_dataset)
        random.shuffle(test_dataset)
        return train_dataset, test_dataset



def get_HCP_Gender(path):
        
        data1 = torch.load(path)
        
        labels = data1['labels']      
        batches = data1['batches']    

        
        label_counts = Counter(labels)
        print("Label counts:", label_counts)

        indices_by_class = defaultdict(list)
        for idx, label in enumerate(labels):
            indices_by_class[label].append(idx)

        train_ratio = 0.8
        train_indices = []
        test_indices = []

        for cls in [0, 1]:
            indices = indices_by_class[cls]
            random.shuffle(indices)
            split_point = int(len(indices) * train_ratio)
            train_indices.extend(indices[:split_point])
            test_indices.extend(indices[split_point:])

        train_dataset = [batches[i] for i in train_indices]
        test_dataset = [batches[i] for i in test_indices]


        random.shuffle(train_dataset)
        random.shuffle(test_dataset)
        return train_dataset, test_dataset