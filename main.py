from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse
from ocr_client import AzureMistralOcrClient
from extraction_agent import ExtractionAgent

app = FastAPI(title="ProyectoIA - Backend", version="1.0.0")

agent = ExtractionAgent()  # usa env vars AZURE_OPENAI_*
ocr_client = AzureMistralOcrClient()  # instancia única


class ExtractRequest(BaseModel):
    ocr_text: str = Field(..., description="Texto plano resultante del OCR (puede ser muy largo).")


@app.on_event("startup")
async def startup() -> None:
    await agent.startup()


@app.on_event("shutdown")
async def shutdown() -> None:
    await agent.shutdown()


@app.get("/")
async def read_root():
    return {"Hello": "World IA!!!"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# -----------------------------
# OCR SOLO (igual a tu endpoint)
# -----------------------------
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
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")


# ---------------------------------------------------
# NUEVO: OCR + EXTRACT INMEDIATO (sin request interno)
# ---------------------------------------------------
@app.post("/ocr/pdf/extract")
async def ocr_pdf_and_extract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Debe enviar un archivo .pdf")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    try:
        # 1) OCR
        ocr_text = await ocr_client.pdf_bytes_to_text(pdf_bytes)

        # 2) EXTRACT inmediatamente (llamada directa, no HTTP)
        extracted = await agent.extract_all(ocr_text)

        return {
            "filename": file.filename,
            "extracted": extracted,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")


# -----------------------------
# EXTRACT (igual a tu endpoint)
# -----------------------------
@app.post("/extract")
async def extract(req: ExtractRequest):
    return await agent.extract_all(req.ocr_text)


@app.post("/extract/stream")
async def extract_stream(req: ExtractRequest):
    return StreamingResponse(agent.stream_extract(req.ocr_text), media_type="text/event-stream")