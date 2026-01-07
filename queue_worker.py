import asyncio
import json
import logging
import base64
import binascii
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, unquote
from azure.storage.queue.aio import QueueClient
from azure.storage.blob.aio import BlobServiceClient

logger = logging.getLogger("queue-worker")


@dataclass
class QueueWorkerConfig:
    storage_connection_string: str
    queue_name: str

    visibility_timeout: int = 600
    max_messages: int = 1

    max_dequeue_before_poison: int = 5

    backoff_initial: float = 1.0
    backoff_max: float = 30.0

    msg_retry_initial: int = 15
    msg_retry_max: int = 300

    poison_queue_name: Optional[str] = None


class QueuePdfProcessor:
    def __init__(self, cfg: QueueWorkerConfig, *, ocr_client, extraction_agent, status_client):
        self.cfg = cfg
        self.ocr_client = ocr_client
        self.agent = extraction_agent
        self.status_client = status_client

        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        self.queue = QueueClient.from_connection_string(
            conn_str=self.cfg.storage_connection_string,
            queue_name=self.cfg.queue_name,
        )

        self.poison_queue: Optional[QueueClient] = None
        if self.cfg.poison_queue_name:
            self.poison_queue = QueueClient.from_connection_string(
                conn_str=self.cfg.storage_connection_string,
                queue_name=self.cfg.poison_queue_name,
            )

        self.blob_service = BlobServiceClient.from_connection_string(
            conn_str=self.cfg.storage_connection_string
        )

    async def start(self) -> None:
        try:
            await self.queue.create_queue()
        except Exception:
            pass

        if self.poison_queue:
            try:
                await self.poison_queue.create_queue()
            except Exception:
                pass

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

        logger.info(
            "Queue worker started. queue=%s poison=%s visibility_timeout=%ss max_messages=%s "
            "max_dequeue=%s loop_backoff=%s..%ss msg_retry=%s..%ss",
            self.cfg.queue_name,
            self.cfg.poison_queue_name,
            self.cfg.visibility_timeout,
            self.cfg.max_messages,
            self.cfg.max_dequeue_before_poison,
            self.cfg.backoff_initial,
            self.cfg.backoff_max,
            self.cfg.msg_retry_initial,
            self.cfg.msg_retry_max,
        )

    async def stop(self) -> None:
        self._stop_event.set()

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        await self.queue.close()
        if self.poison_queue:
            await self.poison_queue.close()
        await self.blob_service.close()

        logger.info("Queue worker stopped.")

    async def _run_loop(self) -> None:
        delay = float(self.cfg.backoff_initial)
        max_delay = float(self.cfg.backoff_max)

        while not self._stop_event.is_set():
            try:
                processed_any = await self.process_one_batch()

                if processed_any:
                    if delay != float(self.cfg.backoff_initial):
                        logger.info("Actividad detectada. Loop backoff reset a %.1fs.", self.cfg.backoff_initial)
                    delay = float(self.cfg.backoff_initial)
                    continue

                logger.info("No hubo mensajes. Durmiendo %.1fs (loop backoff).", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, max_delay)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception("Worker loop error: %s", e)
                logger.info("Error en loop. Durmiendo %.1fs (loop backoff).", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, max_delay)

    async def process_one_batch(self) -> bool:
        messages = self.queue.receive_messages(
            messages_per_page=self.cfg.max_messages,
            visibility_timeout=self.cfg.visibility_timeout,
        )

        got_any = False
        processed_count = 0

        async for msg_page in messages.by_page():
            async for msg in msg_page:
                got_any = True
                processed_count += 1

                ok = await self._handle_message(msg)
                if not ok:
                    await self._apply_message_backoff(msg)

                if processed_count >= self.cfg.max_messages:
                    return True

        return got_any

    def _parse_payload(self, content: str) -> dict:
        if not content:
            raise ValueError("Mensaje vacío")

        try:
            obj = json.loads(content)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        try:
            decoded = base64.b64decode(content).decode("utf-8")
            obj = json.loads(decoded)
            if isinstance(obj, dict):
                return obj
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Mensaje no es JSON ni base64(JSON): {e}")

        raise ValueError("Payload no es dict JSON")

    async def _handle_message(self, msg) -> bool:
        dq = getattr(msg, "dequeue_count", 1) or 1

        # 1) Parsear payload
        try:
            payload = self._parse_payload(msg.content)
            logger.info("Mensaje decodificado OK. dequeue_count=%s keys=%s", dq, list(payload.keys()))
        except Exception as e:
            logger.warning("Payload inválido. Se elimina. err=%s content=%r", e, msg.content)
            await self.queue.delete_message(msg)
            return True

        # 2) Extraer campos del mensaje (tu formato real)
        blob_url = payload.get("blobUrl") or payload.get("blob_url") or payload.get("url")
        id_carga = payload.get("idCarga") or payload.get("id_carga")
        trace_id = payload.get("traceId") or payload.get("trace_id")

        if not blob_url or not id_carga:
            logger.warning("Mensaje sin blobUrl o idCarga. Se elimina. payload=%s", payload)
            await self.queue.delete_message(msg)
            return True

        # 3) Poison (solo cuando se exceden reintentos)
        if dq >= self.cfg.max_dequeue_before_poison:
            try:
                await self.status_client.update_status(
                    id_carga=id_carga,
                    status="ERROR",
                    comment="Max retries exceeded (poison)",
                )
                logger.info("Estado ERROR enviado a Back (poison). idCarga=%s", id_carga)
            except Exception:
                logger.exception("No se pudo actualizar estado ERROR en Back (poison). idCarga=%s", id_carga)

            await self._handle_poison(msg, payload, blob_url, RuntimeError("Max retries exceeded"))
            return True

        # 4) Procesamiento normal
        try:
            pdf_bytes, filename = await self._download_pdf_from_blob_url(blob_url)
            logger.info("PDF descargado OK. idCarga=%s filename=%s size_bytes=%s", id_carga, filename, len(pdf_bytes))

            ocr_text = await self.ocr_client.pdf_bytes_to_text(pdf_bytes)
            logger.info("OCR OK. idCarga=%s filename=%s chars=%s", id_carga, filename, len(ocr_text or ""))

            extracted = await self.agent.extract_all(ocr_text)
            logger.info("AGENTE OK. idCarga=%s filename=%s extracted_keys=%s", id_carga, filename, list(extracted.keys()))

            # Éxito -> PROCESSED
            try:
                await self.status_client.update_status(
                    id_carga=id_carga,
                    status="PROCESSED",
                    comment=f"Procesado OK (traceId={trace_id or 'N/A'})",
                )
                logger.info("Estado PROCESSED enviado a Back. idCarga=%s", id_carga)
            except Exception:
                logger.exception("No se pudo actualizar estado PROCESSED en Back. idCarga=%s", id_carga)

            await self.queue.delete_message(msg)
            logger.info("Mensaje eliminado de la cola. idCarga=%s filename=%s", id_carga, filename)
            return True

        except Exception as e:
            err = str(e)

            # Errores NO transitorios: no vale la pena reintentar
            non_retryable = (
                "Azure OpenAI client not configured" in err
                or "missing env vars" in err
            )

            if non_retryable:
                try:
                    await self.status_client.update_status(
                        id_carga=id_carga,
                        status="ERROR",
                        comment="Fallo IA: Azure OpenAI no configurado (missing env vars)",
                    )
                    logger.info("Estado ERROR enviado a Back (non-retryable). idCarga=%s", id_carga)
                except Exception:
                    logger.exception("No se pudo actualizar estado ERROR en Back (non-retryable). idCarga=%s", id_carga)

                await self.queue.delete_message(msg)
                logger.error("Mensaje eliminado (non-retryable). idCarga=%s", id_carga)
                return True

            # Retry normal
            logger.exception("Error procesando msg. idCarga=%s dequeue_count=%s (retry): %s", id_carga, dq, e)
            return False

    async def _apply_message_backoff(self, msg) -> None:
        dq = getattr(msg, "dequeue_count", 1) or 1
        delay = min(self.cfg.msg_retry_initial * (2 ** max(dq - 1, 0)), self.cfg.msg_retry_max)

        try:
            await self.queue.update_message(msg, visibility_timeout=int(delay))
            logger.info("Backoff por mensaje aplicado: visibility_timeout=%ss dequeue_count=%s", int(delay), dq)
        except Exception:
            logger.exception("No se pudo actualizar visibilidad del mensaje (backoff). dequeue_count=%s", dq)

        await asyncio.sleep(min(2.0, float(delay)))

    async def _handle_poison(self, msg, payload: dict, blob_url: str, error: Exception) -> None:
        try:
            if self.poison_queue:
                poison_payload = {
                    "queue": self.cfg.queue_name,
                    "originalMessageId": getattr(msg, "id", None),
                    "dequeueCount": getattr(msg, "dequeue_count", None),
                    "blobUrl": blob_url,
                    "payload": payload,
                    "error": str(error),
                }
                await self.poison_queue.send_message(json.dumps(poison_payload, ensure_ascii=False))
                logger.error("Mensaje enviado a poison queue=%s blobUrl=%s", self.cfg.poison_queue_name, blob_url)

            await self.queue.delete_message(msg)
            logger.error("Mensaje eliminado definitivamente (poison). blobUrl=%s", blob_url)

        except Exception:
            logger.exception("Fallo manejando poison message. blobUrl=%s", blob_url)

    async def _download_pdf_from_blob_url(self, blob_url: str) -> tuple[bytes, str]:
        u = urlparse(blob_url)
        parts = u.path.lstrip("/").split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"blobUrl inválida: {blob_url}")

        container_name = parts[0]
        blob_name = unquote(parts[1])
        filename = blob_name.split("/")[-1]

        blob = self.blob_service.get_blob_client(container=container_name, blob=blob_name)
        downloader = await blob.download_blob()
        data = await downloader.readall()
        if not data:
            raise ValueError("El blob descargó vacío.")
        return data, filename