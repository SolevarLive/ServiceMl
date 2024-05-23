import streamlit as st
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import nibabel as nib
from torch import nn
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
import tempfile
from skimage.transform import resize
from matplotlib import pyplot as plt

class CustomSegmentationDataset(Dataset):
    
    def __init__(self, im_nii_paths, transformations=None):
        self.transformations = transformations
        self.ims = self.get_slices(im_nii_paths)
        self.n_cls = 3

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):
        im = self.ims[idx]
        if self.transformations:
            im = np.array(im)
            im = self.apply_transformations(im)
        im = self.preprocess_im(im)
        return im.float()

    def preprocess_im(self, im):
        im = torch.clamp(im, min=0)  # Обрезка отрицательных значений, используя torch
        max_val = torch.max(im)
        if max_val > 0:
            im = im / max_val  # Нормализация
        return im 

    def get_slices(self, im_nii_paths):
        ims = []
        nii_im_data = self.read_nii(im_nii_paths)
        for idx, (im) in enumerate(nii_im_data):
            ims.append(im)
        return ims

    def read_nii(self, im_path):
        nii_im_data = nib.load(im_path).get_fdata().transpose(2, 1, 0)
        return nii_im_data

    def apply_transformations(self, im):
        transformed = self.transformations(image=im)
        return transformed["image"]

def load_model(model_path, n_classes=3, in_channels=1):
    model = smp.DeepLabV3Plus(classes=n_classes, in_channels=in_channels)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict(model, data_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for img_batch in data_loader:
            if len(img_batch.size()) == 3:  # Если у нас 3D вход без размерности канала, добавляем канал
                img_batch = img_batch.unsqueeze(1)  # Добавляем измерение канала
            img_batch = img_batch.to(device)
            output = model(img_batch)
            pred_mask = torch.softmax(output, dim=1)
            pred_mask = pred_mask.argmax(dim=1)
            results.extend(pred_mask.cpu().numpy())
    return results

def visualize(img, pred_mask):
    img = img.squeeze() if img.ndim > 2 else img  # Удаление канальной оси, если она присутствует
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img, cmap='bone')
    ax[0].title.set_text('Original Image')
    ax[1].imshow(img, cmap='bone')
    ax[1].imshow(np.ma.masked_where(pred_mask == 0, pred_mask), alpha=0.5, cmap='jet')  
    ax[1].title.set_text('Segmentation')
    plt.show()
    return fig

# Основной блок кода Streamlit
st.title('NIfTI Medical Image Segmentation')

uploaded_file = st.file_uploader("Choose an image...", type="nii", key="file-upload")
MODEL_PATH = 'C:/Users/sulpa/Desktop/model.ran/epoch_49.pth'

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    transformations = A.Compose([
        A.Resize(256, 256),
        ToTensorV2(),
    ])
    dataset = CustomSegmentationDataset(tmp_file_path, transformations=transformations)

    total_slices = len(dataset)
    slice_num = st.slider('Slice', 0, total_slices - 1, 0)

    img_slice = dataset[slice_num][0]  # Получаем тензор из датасета, предварительно преобразованный
    
    if img_slice.ndim == 2:  # Если img_slice имеет два измерения, добавляем измерение канала
        img_slice = img_slice.unsqueeze(0)
    
    
    img_slice = img_slice.unsqueeze(0)

    single_slice_loader = DataLoader([img_slice], batch_size=1)

    model = load_model(MODEL_PATH)
    pred_mask = predict(model, img_slice.unsqueeze(0).to(torch.device('cpu')), torch.device('cpu'))


    fig = visualize(img_slice.squeeze().cpu().numpy(), pred_mask[0])  # Удалить размерность батча для визуализации
    st.pyplot(fig)

    os.remove(tmp_file_path)
