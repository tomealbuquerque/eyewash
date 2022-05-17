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



# Function: Resize images
def resize_images(datapath, newpath, newheight=512):
    
    # Create new directories (if necessary)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    

    # Go through data directory and generate new (resized) images
    for f in tqdm(os.listdir(datapath)):
        if(f.endswith(".jpg") or f.endswith('.png')):
            img = Image.open(os.path.join(datapath, f))
            w, h = img.size
            ratio = w / h
            new_w = int(np.ceil(newheight * ratio))
            new_img = img.resize((new_w, newheight), Image.ANTIALIAS)
            new_img.save(os.path.join(newpath, f))


    return



# StanfordCarsDataset: Dataset Class
class StanfordCarsDataset(Dataset):
    def __init__(self, base_data_path, data_split, resized=False, transform=None):
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
            if resized:
                self.subset_images = os.path.join(self.base_images_path, "resized", "cars_train")

            else:
                self.subset_images = os.path.join(self.base_images_path, "cars_train")
            
            self.annotations_mat_path = os.path.join(self.base_labels_path, "devkit", "cars_train_annos.mat")
        
        else:
            if resized:
                self.subset_images = os.path.join(self.base_images_path, "resized", "cars_test")
                
            else:
                self.subset_images = os.path.join(self.base_images_path, "cars_test")
            
            self.annotations_mat_path = os.path.join(self.base_raw_path, "cars_test_annos_withlabels.mat")


        # Get images and respective labels
        # Note: Original target mapping  starts from 1, hence -1
        self.samples = [(os.path.join(self.subset_images, annotation["fname"]), annotation["class"] - 1) for annotation in loadmat(self.annotations_mat_path, squeeze_me=True)["annotations"]]
        self.class_names = loadmat(os.path.join(self.base_labels_path, "devkit", "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_name_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        self.idx_to_class_name = {i: cls for i, cls in enumerate(self.class_names)}

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
    resize_data = True

    if resize_data:
        for subset_folder in ["cars_test", "cars_train"]:
            datapath = os.path.join("data", "images", subset_folder)
            new_datapath = os.path.join("data", "images", "resized", subset_folder)

            resize_images(datapath=datapath, newpath=new_datapath)

    else:
        for data_split in ["train", "test"]:
            transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            dataset = StanfordCarsDataset(base_data_path="data", data_split=data_split, transform=transforms)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
            

            with torch.no_grad():
                for images, labels in tqdm(dataloader):
                    print(f"Data Split: {data_split}")
                    print(f"Image Shape: {images.shape} | Label: {labels.item()}")
