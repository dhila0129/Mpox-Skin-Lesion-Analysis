import cv2 as cv
import torch
import numpy as np
from os import listdir
from os.path import join
from torch.utils.data import Dataset

class getImageLabel(Dataset):
    def __init__ (self, 
                  augmented = 'd:/Kuliah/Semester 3/IPSD/Deep Learning/Dataset/Augmented Images/Augmented Images/FOLDS_AUG/',
                  original = 'd:/Kuliah/Semester 3/IPSD/Deep Learning/Dataset/Original Images/Original Images/FOLDS/',
                  folds = [1, 2, 3, 4, 5],
                  subdir = ['Train', 'Valid', 'Test']):
        
        self.dataset = []
        to_one_hot = np.eye(6)

        for fold in folds:
            for subdirs in subdir:
                if original:
                    ori = join(original, f"fold{fold}/{subdirs}")
                    for i, pox in enumerate(sorted(listdir(ori))):
                        for image_name in listdir(ori + "/" + pox):
                            image = cv.resize(cv.imread(ori + "/" + pox + "/" + image_name), (32, 32)) / 255
                            self.dataset.append([image, to_one_hot[i]])

                if augmented and subdirs == 'Train':
                    aug = join(augmented, f"fold{fold}_AUG/{subdirs}")
                    for i, pox in enumerate(sorted(listdir(aug))):
                        for image_name in listdir(aug + "/" + pox):
                            image = cv.resize(cv.imread(aug + "/" + pox + "/" + image_name), (32, 32)) / 255
                            self.dataset.append([image, to_one_hot[i]])
        
    def __getitem__(self, item):
        feature, label = self.dataset[item]
        label_index = torch.argmax(torch.tensor(label, dtype=torch.float32))
        return torch.tensor(feature, dtype=torch.float32), label_index
        
    def __len__ (self):
        return len(self.dataset)