import torch.utils.data as data
from torchvision.transforms import transforms
import PIL.Image as Image
from glob import glob

class ConcreteCrackDataset(data.Dataset):
    def __init__(self, state):
        self.state = state
        self.aug = True

        # Crack
        self.train_root = "./Dataset/Concrete_crack/Train"
        self.test_root = "./Dataset/Concrete_crack/Test"

        self.pics,self.masks = self.getDataPath()
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # -> [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) # ->[-1,1]
        self.target_transform = transforms.ToTensor()

    def getDataPath(self):
        assert self.state =='train' or self.state == 'val' or self.state =='test'
        if self.state == 'train':
            root = self.train_root
        if self.state == 'test':
            root = self.test_root

        pics = glob(root + '\images\*')
        masks = glob(root + '\masks\*')
        return pics,masks

    def __getitem__(self, index):
        pic_path = self.pics[index]
        mask_path = self.masks[index]

        pic = self.rgb_loader(pic_path)
        mask = self.binary_loader(mask_path)

        img = self.transform(pic)
        label = self.target_transform(mask)
        
        return img, label, pic_path, mask_path
    
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')
    
    def __len__(self):
        return len(self.pics)