import torch
import clip
from PIL import Image
import numpy as np
from io import BytesIO
import requests

def get_clip_model():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return model, preprocess, None

def preprocess_image(image, preprocess):
    return preprocess(image).unsqueeze(0)

def comparar_con_base(data, model, preprocess, supabase):
    tipo = data.tipoMascota.lower()
    query = supabase.table("publicacionMascota").select("*").eq("tipoMascota", tipo)
    if data.tipoPublicacion == "perdida":
        query = query.eq("tipoPublicacion", "encontrada")
    elif data.tipoPublicacion == "encontrada":
        query = query.eq("tipoPublicacion", "perdida")
    response = query.execute()
    encontrados = response.data if hasattr(response, "data") else response

    similitudes = []

    response = requests.get(data.fotoUrl)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img_input = preprocess(img)
    with torch.no_grad():
        emb_input = model.encode_image(img_input.unsqueeze(0))[0].cpu().numpy()

    for mascota in encontrados:
        emb = mascota.get("embedding")
        if not emb:
            continue
        emb = np.array(emb)
        score_img = float(np.dot(emb_input, emb) / (np.linalg.norm(emb_input) * np.linalg.norm(emb)))

        score = 0.5 * score_img
        if mascota.get("raza") == data.raza:
            score += 0.15
        if mascota.get("sexo") == data.sexo:
            score += 0.1
        if mascota.get("barrio") == data.barrio:
            score += 0.1
        if mascota.get("color").lower() in data.color.lower() or data.color.lower() in mascota.get("color", "").lower():
            score += 0.15

        similitudes.append({
            "id": mascota["id"],
            "foto": mascota.get("fotoUrl"),
            "score": round(score, 4)
        })

    return sorted(similitudes, key=lambda x: x["score"], reverse=True)