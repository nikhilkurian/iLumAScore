import torch
from torch.utils.data import Dataset, DataLoader
import glob
from torchvision import transforms
import random
from PIL import Image
from random import shuffle
# import myTransforms

# Preprocess transformation
# preprocess = myTransforms.HEDJitter(theta=0.05)

class hetero_train(Dataset):
    def __init__(self, path, transforms, pos_class, txt_path):
        self.pos_class = pos_class
        self.path = path
        self.txt_path = txt_path
        self.classes = ['basal', 'her2', 'luma', 'lumb']
        self.transforms = transforms

        patient_dirs = glob.glob(path + '/*/*')
        train_patients = [s.strip() for s in open(self.txt_path)]
        self.train_dirs = [dir_ for dir_ in patient_dirs if dir_.split('/')[-1] in train_patients]
        
        pos_patients = [(s, 1) for s in self.train_dirs if s.split('/')[-2] == pos_class]
        neg_patients = [(s, 0) for s in self.train_dirs if s.split('/')[-2] != pos_class]
        shuffle(neg_patients)
        
        self.train_dirs = pos_patients + neg_patients[:len(pos_patients)]

    def __getitem__(self, index):
        patient, label = self.train_dirs[index]
        patches = glob.glob(patient + '/*')
        patch = random.choice(patches)
        image = Image.open(patch)
        image = preprocess(image)
        image = self.transforms(image)
        return image, label, patch

    def __len__(self):
        return len(self.train_dirs)

class hetero_val(Dataset):
    def __init__(self, path, transforms, pos_class, txt_path):
        self.pos_class = pos_class
        self.txt_path = txt_path
        self.classes = ['basal', 'her2', 'luma', 'lumb']
        self.transforms = transforms

        patient_dirs = glob.glob(path + '/*/*')
        val_patients = [s.strip() for s in open(self.txt_path)]
        val_dirs = [dir_ for dir_ in patient_dirs if dir_.split('/')[-1] in val_patients]
        
        self.samples = []
        for dir_ in val_dirs:
            self.samples += glob.glob(dir_ + '/*')

    def __getitem__(self, index):
        image = Image.open(self.samples[index])
        image = self.transforms(image)
        label = self.classes.index(self.samples[index].split('/')[-3])
        label = 1 if label == self.classes.index(self.pos_class) else 0
        return image, label, self.samples[index]

    def __len__(self):
        return len(self.samples)


class hetero_test(Dataset):
    def __init__(self, transforms, pos_class, txt_path):
        self.pos_class = pos_class
        self.txt_path = txt_path
        self.classes = ['basal', 'her2', 'luma', 'lumb']
        self.transforms = transforms
        self.samples = [s.strip() for s in open(self.txt_path)]


    def __getitem__(self, index):
        image = Image.open(self.samples[index])
        image = self.transforms(image)
        label = self.classes.index(self.samples[index].split('/')[-3])
        label = 1 if label == self.classes.index(self.pos_class) else 0
        return image, label, self.samples[index]

    def __len__(self):
        return len(self.samples)

class TrainRecDataset(Dataset):
    def __init__(self, path, transforms, pos_class, txt_path):
        self.pos_class = pos_class
        self.path = path
        self.txt_path = txt_path
        self.classes = ['basal', 'her2', 'luma', 'lumb']
        self.transforms = transforms

        patient_dirs = glob.glob(path + '/*/*')
        train_patients = [s.strip() for s in open(self.txt_path)]
        self.train_dirs = [dir_ for dir_ in patient_dirs if dir_.split('/')[-1] in train_patients]

        self.all_patches = []
        for patient_dir in self.train_dirs:
            label = 1 if patient_dir.split('/')[-2] == self.pos_class else 0
            patches = glob.glob(patient_dir + '/*')
            for patch in patches:
                self.all_patches.append((patch, label))

    def __getitem__(self, index):
        patch_path, label = self.all_patches[index]
        image = Image.open(patch_path)
        image = self.transforms(image)
        return image, label, patch_path

    def __len__(self):
        return len(self.all_patches)

class MixupTrain(Dataset):
    def __init__(self, path, transforms, pos_class, mixup_patients_path):
        self.pos_class = pos_class
        self.path = path
        self.mixup_patients_path = mixup_patients_path
        self.transforms = transforms

        patient_dirs = glob.glob(path + '/*/*')
        mixup_patients = [s.strip() for s in open(mixup_patients_path)]
        
        pos_mixup_patients = [(s, 1) for s in patient_dirs if s.split('/')[-2] == self.pos_class and s.split('/')[-1] in mixup_patients]
        neg_mixup_patients = [(s, 0) for s in patient_dirs if s.split('/')[-2] != self.pos_class and s.split('/')[-1] in mixup_patients]
        
        self.train_dirs = pos_mixup_patients + neg_mixup_patients

    def __getitem__(self, index):
        patient, label = self.train_dirs[index]
        patches = glob.glob(patient + '/*')
        patch = random.choice(patches)
        image = Image.open(patch)
        image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.train_dirs)