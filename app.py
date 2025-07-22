from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import os
from dotenv import load_dotenv
from supabase import create_client
from utils import comparar_con_base, get_clip_model, preprocess_image

load_dotenv()

app = FastAPI()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE = os.getenv("SUPABASE_SERVICE_ROLE")
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE:
    raise Exception("Faltan variables de entorno necesarias.")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)

model, preprocess, tokenizer = get_clip_model()

class ImagenRequest(BaseModel):
    url: str

class ComparacionRequest(BaseModel):
    fotoUrl: str
    tipoMascota: str
    raza: str
    sexo: str
    barrio: str
    color: str
    tipoPublicacion: str

@app.post("/embedding")
def obtener_embedding(data: ImagenRequest):
    try:
        response = requests.get(data.url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image_input = preprocess_image(image, preprocess)
        with torch.no_grad():
            embedding = model.encode_image(image_input)[0].cpu().numpy()
        return embedding.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/comparar-publicacion")
def comparar(data: ComparacionRequest):
    try:
        resultados = comparar_con_base(data, model, preprocess, supabase)
        return resultados
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))