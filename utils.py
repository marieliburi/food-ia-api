from deep_translator import GoogleTranslator
from unidecode import unidecode
from supabase import create_client, Client
import cv2
import os

# ======================
# 🔑 CONFIGURAÇÃO DO SUPABASE
# ======================
SUPABASE_URL = "https://tawiuodyookyckzfbfdt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRhd2l1b2R5b29reWNremZiZmR0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NDQ2Njc4NiwiZXhwIjoyMDYwMDQyNzg2fQ.IE9l5HYmWtPtuS9M1y72z_KE2C8DDH83SaF4hY4d2qA"
TABLE_NAME = "alimentos"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ======================
# 🧩 MAPEAMENTO DE SINÔNIMOS
# ======================
sinonimos = {
    "french fries": "batata frita",
    "pork": "carne de porco",
    "french beans": "vagem",
    "beans": "feijão",
    "rice": "arroz",
    "sauce": "molho",
    "carrot": "cenoura",
    "potato": "batata",
    "chicken": "frango",
    "fish": "peixe",
    "egg": "ovo",
    "beef": "carne bovina",
    "background": "fundo"
}

# ======================
# 🧠 FUNÇÃO PRINCIPAL
# ======================
def process_image(model, image_path):
    """
    Executa o YOLO, traduz nomes e busca calorias no Supabase.
    """
    results = model(image_path)
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    total_kcal = 0
    alimentos_detectados = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label_en = result.names[cls_id].lower()

            # Ignora fundo
            if label_en in ["background", "fundo"]:
                continue

            # Corrige/traduz nome
            label_pt = sinonimos.get(
                label_en,
                GoogleTranslator(source="en", target="pt").translate(label_en)
            )

            termos = unidecode(label_pt.lower().strip())
            palavras = termos.split()

            # Estima peso
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            proporcao = area / (width * height)
            peso_estimado = proporcao * 300  # peso total estimado do prato

            # 🔍 Busca alimento no Supabase
            calorias_100g = 0.0
            nome_banco = "Não encontrado"

            for palavra in palavras:
                query = (
                    supabase.table(TABLE_NAME)
                    .select("*")
                    .ilike("nome_alimento", f"%{palavra}%")
                    .execute()
                )

                if query.data:
                    item = query.data[0]
                    calorias_100g = float(item.get("kcal", 0))
                    nome_banco = item["nome_alimento"]
                    break

            # Calcula calorias
            calorias_estimadas = (peso_estimado / 100) * calorias_100g
            total_kcal += calorias_estimadas

            alimentos_detectados.append({
                "alimento_detectado": label_pt,
                "nome_banco": nome_banco,
                "peso_estimado_g": round(peso_estimado, 1),
                "calorias_estimadas": round(calorias_estimadas, 1)
            })

    return {
        "alimentos": alimentos_detectados,
        "total_kcal": round(total_kcal, 1)
    }
