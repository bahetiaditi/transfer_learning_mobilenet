# train_feature_extraction.py
from models.segnet import SegNet
from utils.dataset import ISICDataset
from utils.metrics import calc_iou, calc_dice
import matplotlib.pyplot as plt
import torch


image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])



train_image_dir = "/kaggle/input/isic-2016/ISIC 2016/train"
train_mask_dir = "/kaggle/input/isic-2016/ISIC 2016/train_masks"
test_image_dir = '/kaggle/input/isic-2016/ISIC 2016/test'
test_mask_dir = '/kaggle/input/isic-2016/ISIC 2016/test_masks'


full_train_dataset = ISICDataset(train_image_dir, train_mask_dir, image_transform=image_transform,mask_transform=mask_transform)


train_indices, val_indices = train_test_split(
    np.arange(len(full_train_dataset)),
    test_size=0.2,
    random_state=42,
    shuffle=True
)


from torch.utils.data import SubsetRandomSampler

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(full_train_dataset, batch_size=16, sampler=train_sampler)
val_loader = DataLoader(full_train_dataset, batch_size=16, sampler=val_sampler)


test_dataset = ISICDataset(test_image_dir, test_mask_dir, image_transform=image_transform,mask_transform=mask_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
mobilenet_v2 = models.mobilenet_v2(pretrained=True).features
for param in mobilenet_v2.parameters():
    param.requires_grad = False


decoder = Decoder()
segnet = SegNet(mobilenet_v2, decoder)


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, segnet.parameters()), lr=0.001)

train_and_validate(segnet, criterion, optimizer, train_loader,val_loader, num_epochs=10)

segnet_2 = SegNet(mobilenet_v2, decoder)

criterion_2 = IoULoss()
optimizer_2 = optim.Adam(filter(lambda p: p.requires_grad, segnet.parameters()), lr=0.0001)

train_and_validate(segnet_2, criterion_2, optimizer_2, train_loader,val_loader, num_epochs=15)
test_model(segnet_2, criterion_2, test_loader)
