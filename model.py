import torch
from torchmetrics import JaccardIndex
from torchvision.models.segmentation import deeplabv3_resnet50
import streamlit as st

def prediction(images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preds = []

    model = deeplabv3_resnet50(num_classes=3, weights=None)
    model.backbone.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=1)
    
    checkpoint = torch.load('deeplabv3.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])

    model.eval()

    progress_bar = st.progress(0)
    st.caption('Обработка')
    total_images = len(images)

    with torch.no_grad():
        for i, image in enumerate(images):
            image = torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
            pred = model(image)['out'].to(device)
            pred = torch.argmax(pred, dim=1)
            preds.append(pred)

            progress_bar.progress((i + 1) / total_images)

    st.success('Предсказание завершено')

    return preds

def iou(preds, masks):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    jaccard = []
    jaccard_metric = JaccardIndex(task='multiclass', num_classes=3).to(device)

    for pred, mask in zip(preds, masks):
        
        mask = torch.tensor(mask).unsqueeze(0).to(device)

        jaccard_metric(pred, mask)
        jaccard.append(jaccard_metric.compute().item())

    # Очистка состояния метрики
    jaccard_metric.reset()

    return sum(jaccard) / len(jaccard)
