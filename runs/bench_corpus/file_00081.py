from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import config


class MapDataset(Dataset):
    def __init__(self, root_dir, reverse=True):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.list_files_flipped = []
        for file in self.list_files:
            self.list_files_flipped.append(file)
            if reverse:
                self.list_files_flipped.append('rev_' + file)

        self.list_files = self.list_files_flipped
        print(self.list_files)

    def __len__(self):
        return len(self.list_files)

    # one image is 1200x800 => 2400x800 both of them
    def __getitem__(self, index):
        img_file = self.list_files[index]  # take the image
        reverse_image = False
        if img_file.startswith('rev_'):  # reverse the image
            reverse_image = True
            img_file = img_file.replace('rev_', '')
        img_path = os.path.join(self.root_dir, img_file)  # take the image path
        image = np.array(Image.open(img_path))  # open the image -> to np array
        if image.shape[2] == 4:
            # drop the alpha channels (if present)
            image = image[:, :, :3]
        input_image = image[:, 1200:, :]  # split the image into two
        target_image = image[:, :1200, :]

        # apply augmentations (resize)
        augmentations = config.both_transform(image=input_image, image0=target_image)
        input_image, target_image = augmentations["image"], augmentations["image0"]
        if reverse_image:
            input_image, target_image = config.flip_images(input_image, target_image)

        input_image = config.transform_only_input(image=input_image)["image"]
        target_image = config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image
