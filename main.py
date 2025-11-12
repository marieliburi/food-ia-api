from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from utils import process_image
import uvicorn
import os

app = FastAPI(title="🍽️ API de Reconhecimento de Alimentos")

# Caminho do modelo (coloque o seu .pt na pasta /model)
MODEL_PATH = "model/best.pt"
model = YOLO(MODEL_PATH)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recebe uma imagem de prato e retorna os alimentos detectados com calorias estimadas.
    """
    try:
        # Salva temporariamente a imagem
        image_path = f"temp_{file.filename}"
        with open(image_path, "wb") as f:
            f.write(await file.read())

        # Processa a imagem e obtém os resultados
        results = process_image(model, image_path)

        # Remove imagem temporária
        os.remove(image_path)

        return JSONResponse(content={"status": "success", "data": results})

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)})


@app.get("/")
def home():
    return {"message": "API de Reconhecimento de Alimentos 🍴 está online!"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
