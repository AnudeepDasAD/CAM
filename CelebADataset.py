from PIL import Image
from torch.utils.data import Dataset

# have to put in separate file from celeba_training.ipynb due to multiprocessing issues
class MultiClassCelebA(Dataset):
    def __init__(self, dataframe, folder_dir, transform=None):
        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.transform = transform
        self.file_names = dataframe.index
        self.labels = dataframe.labels.values.tolist()
        
        
    def __len__(self):
        return len(self.dataframe)
    
    
    def __getitem__(self, index):
        image = Image.open(f'{self.folder_dir}/{self.file_names[index]}')
        label = self.labels[index][0]
        sample = {'image': image, 'label': label.astype(float)}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image, 'label': label.astype(float)}
        
        return sample
