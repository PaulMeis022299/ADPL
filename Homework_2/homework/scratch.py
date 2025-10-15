'''
from data import ImageDataset
import torch

dataset = ImageDataset("train")

def train_dataloader():
            dataset = ImageDataset("train")
            return torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)


trainer = train_dataloader()
print(len(trainer))
'''

from tqdm import tqdm
import zipfile

with zipfile.ZipFile("supertux_data.zip", 'r') as zip_ref:
    files = zip_ref.namelist()
    for file in tqdm(files, desc="Unzipping files"):
        zip_ref.extract(file, "supertux_data")