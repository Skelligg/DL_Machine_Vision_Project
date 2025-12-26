import io
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms

from app.model import FashionCNN, CLASS_NAMES

app = FastAPI(title="Fashion Classifier", description="Upload an image to classify clothing items")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FashionCNN(num_classes=10)
checkpoint_path = Path(__file__).resolve().parent.parent / "models" / "model_checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "healthy", "model": "FashionCNN", "accuracy": "90.65%"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence_score = confidence.item() * 100

    all_probs = {CLASS_NAMES[i]: round(probabilities[0][i].item() * 100, 2) for i in range(10)}
    sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))

    return {
        "prediction": predicted_class,
        "confidence": f"{confidence_score:.2f}%",
        "all_probabilities": sorted_probs
    }
