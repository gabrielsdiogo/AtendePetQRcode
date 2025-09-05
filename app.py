# app.py
import base64
import io
from typing import List, Optional, Literal

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
from PIL import Image

app = FastAPI(title="QR API", version="1.0.1")

# ---------------------- Utils: leitura ----------------------

def _strip_data_url(b64: str) -> str:
    """Remove prefixo data URL se existir."""
    return b64.split(",", 1)[1] if "," in b64 else b64

def _decode_base64_to_cv_image(b64: str):
    """Converte base64 (data URL ou puro) em imagem OpenCV (BGR)."""
    try:
        img_bytes = base64.b64decode(_strip_data_url(b64))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"base64 inválido: {e}")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Falha ao decodificar a imagem.")
    return img

def ler_qrcodes_base64(base64_str: str) -> List[str]:
    """Lê 1+ QR codes de uma imagem base64. Compatível com variações do OpenCV."""
    img = _decode_base64_to_cv_image(base64_str)
    detector = cv2.QRCodeDetector()
    resultados: List[str] = []

    # detectAndDecodeMulti:
    # - Novas versões: (ok, dados, pontos, straight)
    # - Antigas:       (dados, pontos, straight)
    try:
        res = detector.detectAndDecodeMulti(img)
        if isinstance(res, tuple):
            if len(res) == 4:
                ok, dados, _pontos, _straight = res
                if ok and dados:
                    resultados.extend([d for d in (dados or []) if d])
            elif len(res) == 3:
                dados, _pontos, _straight = res
                if dados:
                    resultados.extend([d for d in (dados or []) if d])
    except Exception:
        pass

    # Fallback: detectAndDecode
    # - Pode retornar (texto, pontos, straight) ou (texto, pontos)
    if not resultados:
        try:
            res1 = detector.detectAndDecode(img)
            if isinstance(res1, tuple):
                texto = res1[0] if len(res1) >= 1 else None
                if texto:
                    resultados.append(texto)
            else:
                if res1:
                    resultados.append(res1)
        except Exception:
            pass

    return resultados

# ---------------------- Utils: geração ----------------------

EC_MAP = {
    "L": ERROR_CORRECT_L,
    "M": ERROR_CORRECT_M,
    "Q": ERROR_CORRECT_Q,
    "H": ERROR_CORRECT_H,
}

def gerar_qr_pil(
    texto: str,
    ec_level: str = "M",
    box_size: int = 10,
    border: int = 4,
    fill_color: str = "black",
    back_color: str = "white",
) -> Image.Image:
    if not texto:
        raise HTTPException(status_code=400, detail="Conteúdo do QR vazio.")

    qr = qrcode.QRCode(
        version=None,  # escolhe automaticamente o menor que caiba
        error_correction=EC_MAP.get(ec_level.upper(), ERROR_CORRECT_L),
        box_size=box_size,
        border=border,
    )
    qr.add_data(texto)
    qr.make(fit=True)
    img = qr.make_image(fill_color=fill_color, back_color=back_color)
    if not isinstance(img, Image.Image):
        img = img.convert("RGB")
    return img

def pil_png_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

# ---------------------- Schemas ----------------------

class ReadRequest(BaseModel):
    image_base64: str = Field(..., description="Imagem em base64 (pode ser data URL).")

class ReadResponse(BaseModel):
    results: List[str]

class GenerateRequest(BaseModel):
    content: str = Field(..., description="Conteúdo a ser codificado no QR.")
    ec: Literal["L", "M", "Q", "H"] = "M"  # Pydantic v2-friendly
    box: int = Field(10, ge=1, le=50, description="Tamanho do módulo.")
    border: int = Field(4, ge=0, le=20, description="Borda em módulos.")
    return_base64: bool = Field(True, description="Se true, retorna PNG em base64 (data URL).")

class GenerateResponse(BaseModel):
    data_url_png: Optional[str] = None

# ---------------------- Endpoints ----------------------

@app.post("/read", response_model=ReadResponse, tags=["qr"])
def read_qr(req: ReadRequest):
    textos = ler_qrcodes_base64(req.image_base64)
    return ReadResponse(results=textos)

@app.post("/generate", response_model=GenerateResponse, tags=["qr"])
def generate_qr(req: GenerateRequest):
    img = gerar_qr_pil(
        texto=req.content,
        ec_level=req.ec,
        box_size=req.box,
        border=req.border,
        fill_color="black",
        back_color="white",
    )
    data_url = pil_png_to_data_url(img) if req.return_base64 else None
    return GenerateResponse(data_url_png=data_url)

@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}

# Execução direta (útil fora do Docker)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
