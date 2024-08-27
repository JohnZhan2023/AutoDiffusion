import os
from datasets.arrow_dataset import Dataset


def load_dataset(root):
    datasets = []
    for dataset in os.listdir(root):
        dataset_path = os.path.join(root, dataset)
        dataset = Dataset.load_from_disk(dataset_path)
        datasets.extend(dataset)
    return datasets