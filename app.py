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
from pyzbar.pyzbar import decode as zbar_decode

app = FastAPI(title="QR/Barcode API", version="1.3.0")

# ---------------------- Utils: decodificação base64 ----------------------

def _strip_data_url(b64: str) -> str:
    """Remove prefixo data URL se existir."""
    return b64.split(",", 1)[1] if "," in b64 else b64

def _decode_base64_to_cv_and_pil(b64: str):
    """Converte base64 (data URL ou puro) em (OpenCV, PIL)."""
    try:
        img_bytes = base64.b64decode(_strip_data_url(b64))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"base64 inválido: {e}")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise HTTPException(status_code=400, detail="Falha ao decodificar imagem (OpenCV).")

    try:
        img_pil = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Falha ao abrir imagem (PIL): {e}")

    return img_cv, img_pil

# ---------------------- Leitura: QR + Barras ----------------------

def ler_codigos_base64(base64_str: str) -> List[str]:
    """Lê QR codes e barcodes 1D e retorna apenas os dados (strings)."""
    img_cv, img_pil = _decode_base64_to_cv_and_pil(base64_str)

    resultados: List[str] = []
    vistos = set()

    detector = cv2.QRCodeDetector()

    # --- QR com OpenCV ---
    try:
        res = detector.detectAndDecodeMulti(img_cv)
        if isinstance(res, tuple):
            if len(res) == 4:
                ok, dados, *_ = res
                if ok and dados:
                    for d in (dados or []):
                        if d and d not in vistos:
                            vistos.add(d)
                            resultados.append(d)
            elif len(res) == 3:
                dados, *_ = res
                if dados:
                    for d in (dados or []):
                        if d and d not in vistos:
                            vistos.add(d)
                            resultados.append(d)
    except Exception:
        pass

    if not resultados:
        try:
            res1 = detector.detectAndDecode(img_cv)
            if isinstance(res1, tuple):
                texto = res1[0] if len(res1) >= 1 else None
                if texto and texto not in vistos:
                    vistos.add(texto)
                    resultados.append(texto)
            else:
                if res1 and res1 not in vistos:
                    vistos.add(res1)
                    resultados.append(res1)
        except Exception:
            pass

    # --- Barcodes com pyzbar (inclui QR também) ---
    try:
        for obj in zbar_decode(img_pil):
            try:
                d = obj.data.decode("utf-8", errors="strict")
            except Exception:
                d = obj.data.decode("latin-1", errors="replace")
            if d and d not in vistos:
                vistos.add(d)
                resultados.append(d)
    except Exception:
        # Se libzbar não estiver instalada, não quebra a API; apenas não adiciona nada
        pass

    return resultados

# ---------------------- Geração de QR ----------------------

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
        version=None,
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
    ec: Literal["L", "M", "Q", "H"] = "M"
    box: int = Field(10, ge=1, le=50, description="Tamanho do módulo.")
    border: int = Field(4, ge=0, le=20, description="Borda em módulos.")
    return_base64: bool = Field(True, description="Se true, retorna PNG em base64 (data URL).")

class GenerateResponse(BaseModel):
    data_url_png: Optional[str] = None

# ---------------------- Endpoints ----------------------

@app.post("/read", response_model=ReadResponse, tags=["read"])
def read_codes(req: ReadRequest):
    textos = ler_codigos_base64(req.image_base64)
    return ReadResponse(results=textos)

@app.post("/generate", response_model=GenerateResponse, tags=["generate"])
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
