# Imports
import os
import numpy as np
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm


# PyTorch Imports
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision



# StanfordCarsDataset: Dataset Class
class StanfordCarsDataset(Dataset):
    def __init__(self, base_data_path, data_split, transform=None):
        """
        Args:
            base_data_path (string): Data directory.
            data_split (string): Data split.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        assert data_split.lower() in ("train".lower(), "test".lower()), f"You should provide a valid data split ('train', 'test')."

        # Init variables
        self.base_data_path = base_data_path
        self.base_images_path = os.path.join(base_data_path, "images")
        self.base_labels_path = os.path.join(base_data_path, "labels")
        self.base_raw_path = os.path.join(base_data_path, "raw")
        self.data_split = data_split

        
        # Train Split Variables
        if self.data_split.lower() == "train".lower():
            self.annotations_mat_path = os.path.join(self.base_labels_path, "devkit", "cars_train_annos.mat")
            self.subset_images = os.path.join(self.base_images_path, "cars_train")
        
        else:
            self.annotations_mat_path = os.path.join(self.base_raw_path, "cars_test_annos_withlabels.mat")
            self.subset_images = os.path.join(self.base_images_path, "cars_test")


        # Get images and respective labels
        # Note: Original target mapping  starts from 1, hence -1
        self.samples = [(os.path.join(self.subset_images, annotation["fname"]), annotation["class"] - 1) for annotation in loadmat(self.annotations_mat_path, squeeze_me=True)["annotations"]]
        self.class_names = loadmat(os.path.join(self.base_labels_path, "devkit", "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_name_to_idx = {cls: i for i, cls in enumerate(self.class_names)}        

        # Transform
        self.transform = transform

        return


    # Method: __len__
    def __len__(self):
        return len(self.samples)


    # Method: __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()


        # Get image paths and labels
        image_path, target = self.samples[idx]
        pil_image = Image.open(image_path).convert("RGB")


        # Transform images if needed
        if self.transform:
            pil_image = self.transform(pil_image)
        


        
        return pil_image, target



# To run and test the code
if __name__ == "__main__":
    for data_split in ["train", "test"]:
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        dataset = StanfordCarsDataset(base_data_path="data", data_split=data_split, transform=transforms)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        

        with torch.no_grad():
            for images, labels in tqdm(dataloader):
                print(f"Data Split: {data_split}")
                print(f"Image Shape: {images.shape} | Label: {labels.item()}")