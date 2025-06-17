from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from io import BytesIO

app = FastAPI()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class Imagenes(BaseModel):
    url1: str
    url2: str

@app.post("/comparar")
async def comparar_imagenes(data: Imagenes):
    try:
        img1 = Image.open(BytesIO(requests.get(data.url1).content)).convert("RGB")
        img2 = Image.open(BytesIO(requests.get(data.url2).content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudieron descargar las imágenes.")

    inputs = processor(images=[img1, img2], return_tensors="pt")
    embeddings = model.get_image_features(**inputs)
    similarity = torch.nn.functional.cosine_similarity(
        embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)
    ).item()

    return {"similitud": round(similarity, 4)}
