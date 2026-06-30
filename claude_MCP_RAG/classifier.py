from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import cv2
import timm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

PEST_LABELS = ["정상", "담배가루이 성충", "담배가루이 유충", "애못털진딧물"]
NUM_CLASSES = 4

_BASE = Path(__file__).parent.parent / "pest_classification" / "output"
VIT_MODEL_PATH = _BASE / "vit-aug" / "vit_base_12.pth"
CNN_MODEL_PATH = _BASE / "cnn-aug" / "cnn_base_44.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_transform = A.Compose([A.Resize(224, 224), ToTensorV2()])


class InsectModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


class CustomConvNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.layer1 = self._conv_module(3, 16)
        self.layer2 = self._conv_module(16, 32)
        self.layer3 = self._conv_module(32, 64)
        self.layer4 = self._conv_module(64, 128)
        self.layer5 = self._conv_module(128, 256)
        self.gap = self._global_avg_pool(256, num_classes)

    def forward(self, x):
        out = self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x)))))
        return self.gap(out).view(-1, self.num_classes)

    def _conv_module(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
        )

    def _global_avg_pool(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )


_vit_model: InsectModel | None = None
_cnn_model: CustomConvNet | None = None


def _load_vit() -> InsectModel:
    global _vit_model
    if _vit_model is None:
        m = InsectModel(NUM_CLASSES)
        m.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=device))
        m.to(device).eval()
        _vit_model = m
    return _vit_model


def _load_cnn() -> CustomConvNet:
    global _cnn_model
    if _cnn_model is None:
        m = CustomConvNet(NUM_CLASSES)
        m.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
        m.to(device).eval()
        _cnn_model = m
    return _cnn_model


UPLOADS_DIR = Path(__file__).parent / "uploads"
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def get_latest_upload() -> Optional[Path]:
    """Return the most recently modified image file in the uploads folder, or None."""
    candidates = [p for p in UPLOADS_DIR.iterdir() if p.suffix.lower() in _IMAGE_EXTS]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_model(img_bgr: np.ndarray, model_type: str) -> dict:
    """Core inference on a BGR numpy array."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.as_tensor(
        _transform(image=rgb)["image"], dtype=torch.float32
    ).unsqueeze(0).to(device)

    model = _load_vit() if model_type.lower() == "vit" else _load_cnn()

    with torch.no_grad():
        probs = model(tensor).softmax(1)[0].cpu().numpy()

    predicted_idx = int(probs.argmax())
    return {
        "predicted_class": PEST_LABELS[predicted_idx],
        "confidence": round(float(probs[predicted_idx]), 4),
        "probabilities": {PEST_LABELS[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)},
        "model_used": model_type.upper(),
    }



def classify_image(image_path: Optional[str] = None, model_type: str = "vit") -> dict:
    """Run pest classification on an image file. Returns label, confidence, and per-class probs."""
    if image_path is None:
        latest = get_latest_upload()
        if latest is None:
            raise ValueError(
                f"uploads/ 폴더({UPLOADS_DIR})에 이미지 파일이 없습니다. "
                "이미지를 해당 폴더에 저장한 뒤 다시 시도하세요."
            )
        image_path = str(latest)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")
    return _run_model(img, model_type)
