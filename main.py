# main.py
import os
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse

from ocr_client import AzureMistralOcrClient
from extraction_agent import ExtractionAgent
from queue_worker import QueueWorkerConfig, QueuePdfProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Opcional: bajar ruido del SDK de Azure
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.storage").setLevel(logging.WARNING)

app = FastAPI(title="ProyectoIA - Backend", version="1.0.0")

ocr_client = AzureMistralOcrClient()
agent = ExtractionAgent()

worker: QueuePdfProcessor | None = None


@app.on_event("startup")
async def startup() -> None:
    await agent.startup()

    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    queue_name = os.getenv("AZURE_STORAGE_QUEUE_NAME", "").strip()

    # Visibilidad / batch
    visibility_timeout = int(os.getenv("QUEUE_VISIBILITY_TIMEOUT", "600"))
    max_messages = int(os.getenv("QUEUE_MAX_MESSAGES", "1"))

    # Poison / retries
    max_dequeue_before_poison = int(os.getenv("QUEUE_MAX_DEQUEUE_BEFORE_POISON", "5"))
    poison_queue_name = os.getenv("QUEUE_POISON_NAME", "").strip() or None

    # Backoff loop (cola vacía)
    backoff_initial = float(os.getenv("QUEUE_BACKOFF_INITIAL", "1"))
    backoff_max = float(os.getenv("QUEUE_BACKOFF_MAX", "30"))

    # Backoff por mensaje (cuando falla procesamiento)
    msg_retry_initial = int(os.getenv("QUEUE_MSG_RETRY_INITIAL", "15"))
    msg_retry_max = int(os.getenv("QUEUE_MSG_RETRY_MAX", "300"))

    if not conn:
        logger.warning("AZURE_STORAGE_CONNECTION_STRING no configurada. Worker NO iniciará.")
        return

    global worker
    worker = QueuePdfProcessor(
        QueueWorkerConfig(
            storage_connection_string=conn,
            queue_name=queue_name,
            visibility_timeout=visibility_timeout,
            max_messages=max_messages,
            max_dequeue_before_poison=max_dequeue_before_poison,
            poison_queue_name=poison_queue_name,
            backoff_initial=backoff_initial,
            backoff_max=backoff_max,
            msg_retry_initial=msg_retry_initial,
            msg_retry_max=msg_retry_max,
        ),
        ocr_client=ocr_client,
        extraction_agent=agent,
    )

    await worker.start()

    logger.info(
        "Worker iniciado. queue=%s visibility_timeout=%s max_messages=%s max_dequeue=%s poison=%s "
        "loop_backoff=%s..%s msg_retry=%s..%s",
        queue_name,
        visibility_timeout,
        max_messages,
        max_dequeue_before_poison,
        poison_queue_name,
        backoff_initial,
        backoff_max,
        msg_retry_initial,
        msg_retry_max,
    )


@app.on_event("shutdown")
async def shutdown() -> None:
    global worker
    if worker:
        await worker.stop()
    await agent.shutdown()


@app.get("/")
async def read_root():
    return {"Hello": "World IA!!!"}


@app.get("/health")
async def health():
    return {"status": "ok"}


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


@app.post("/ocr/pdf/extract")
async def ocr_pdf_and_extract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Debe enviar un archivo .pdf")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacío")

    try:
        ocr_text = await ocr_client.pdf_bytes_to_text(pdf_bytes)
        extracted = await agent.extract_all(ocr_text)

        return JSONResponse(
            status_code=200,
            content={
                "filename": file.filename,
                "extracted": extracted,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {e}")


@app.post("/worker/process-once")
async def worker_process_once():
    global worker
    if not worker:
        raise HTTPException(status_code=500, detail="Worker no está configurado.")
    processed = await worker.process_one_batch()
    return {"processed": processed}