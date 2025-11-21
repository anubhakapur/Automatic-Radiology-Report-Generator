import torch
import timm
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def get_vit_model(device='cpu'):
    """
    Loads pretrained ViT model.
    """
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model.eval()
    model.to(device)
    return model


def get_preprocessing_transform():
    """
    Standard ViT image preprocessing.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

def extract_features_single(model, image, device='cpu'):
    preprocess = get_preprocessing_transform()
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model.forward_features(img_tensor)

    return feat.cpu().numpy().flatten()

def extract_features(model, image_paths, device='cpu', batch_size=16, save_path=None):
    preprocess = get_preprocessing_transform()
    features = []

    valid_paths = []
    for p in image_paths:
        try:
            Image.open(p)  # test open
            valid_paths.append(p)
        except:
            print(f"[WARNING] Skipping missing or corrupted file: {p}")

    for i in tqdm(range(0, len(valid_paths), batch_size)):
        batch_paths = valid_paths[i:i+batch_size]
        batch_imgs = []

        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                batch_imgs.append(preprocess(img))
            except:
                print(f"[WARNING] Error reading image, skipping: {path}")
                continue

        if len(batch_imgs) == 0:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)

        with torch.no_grad():
            batch_feat = model.forward_features(batch_tensor)

        features.append(batch_feat.cpu().numpy())

    if len(features) == 0:
        print("❌ No valid images found. Feature extraction aborted.")
        return np.array([])

    features = np.concatenate(features, axis=0)

    if save_path:
        np.save(save_path, features)
        print(f"✅ Saved features to {save_path}")

    return features
