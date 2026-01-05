from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse

from ocr_client import AzureMistralOcrClient

app = FastAPI()

# Instancia única del cliente
ocr_client = AzureMistralOcrClient()


@app.get("/")
async def read_root():
    return {"Hello": "World IA!!!"}


@app.post("/ocr/pdf", response_class=PlainTextResponse)
async def ocr_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Debe enviar un archivo .pdf")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    try:
        text = await ocr_client.pdf_bytes_to_text(pdf_bytes)
        return text
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # Error upstream OCR
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")
