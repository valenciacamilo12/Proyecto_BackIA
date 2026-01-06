import os
import base64
import asyncio
import random
import logging
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, Iterable

import httpx
from pypdf import PdfReader, PdfWriter

logger = logging.getLogger("ocr-client")


@dataclass
class OcrChunkingConfig:
    # Límite duro por páginas (seguro para el servicio)
    max_pages_per_request: int = 30

    # Límite por tamaño del PDF chunk (reduce 500 por payload grande)
    # Recomendado: 4 a 8 MB dependiendo del servicio y estabilidad.
    target_chunk_max_bytes: int = 6 * 1024 * 1024  # 6MB

    # Si un chunk supera el límite, intentamos partirlo más (hasta 1 página).
    min_pages_per_chunk: int = 1


class AzureMistralOcrClient:
    """
    OCR 100% por API. pypdf SOLO para:
    - contar páginas
    - dividir PDF en chunks válidos

    Estrategia:
    - Preferir OCR por URL (barato, rápido, menos 500).
    - Si no se puede (403/404 o url no usable), fallback a bytes/base64.
    - Para PDFs grandes, dividir en chunks adaptativos (páginas y bytes).
    - Reintentos con backoff+jitter para 429/5xx y timeouts.
    """

    RETRYABLE_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
        ocr_url: str = "https://servicios.services.ai.azure.com/providers/mistral/azure/ocr",
        model: str = "mistral-document-ai-2505",
        timeout_seconds: float = 240.0,
        connect_timeout_seconds: float = 30.0,
        include_image_base64: bool = False,
        add_part_headers: bool = True,
        # Retries
        max_retries: int = 6,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 20.0,
        # Chunking
        chunking: Optional[OcrChunkingConfig] = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("AZURE_MISTRAL_API_KEY", "")).strip()
        self.ocr_url = ocr_url
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)
        self.include_image_base64 = include_image_base64
        self.add_part_headers = add_part_headers

        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds

        self.chunking = chunking or OcrChunkingConfig()

        self._client: Optional[httpx.AsyncClient] = None

    async def startup(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict:
        if not self.api_key:
            raise ValueError("Falta la API Key. Configure AZURE_MISTRAL_API_KEY.")
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    @staticmethod
    def _extract_text_defensive(data: dict) -> str:
        for key in ("text", "output_text", "content", "result"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        texts: list[str] = []

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in ("text", "content", "markdown") and isinstance(v, str) and v.strip():
                        texts.append(v.strip())
                    walk(v)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it)

        walk(data)

        seen = set()
        uniq = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return "\n".join(uniq).strip()

    async def _post_with_retries(self, payload: dict) -> httpx.Response:
        if self._client is None:
            await self.startup()
        assert self._client is not None

        last_resp: Optional[httpx.Response] = None
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = await self._client.post(self.ocr_url, headers=self._headers(), json=payload)
                last_resp = resp

                if resp.status_code < 400:
                    return resp

                if resp.status_code in self.RETRYABLE_STATUS:
                    sleep_s = min(self.retry_max_seconds, self.retry_base_seconds * (2 ** (attempt - 1)))
                    sleep_s *= (0.7 + random.random() * 0.6)  # jitter
                    logger.warning(
                        "OCR retryable status=%s attempt=%s/%s sleeping=%.2fs",
                        resp.status_code, attempt, self.max_retries, sleep_s
                    )
                    await asyncio.sleep(sleep_s)
                    continue

                return resp  # error no reintentable

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as e:
                last_exc = e
                sleep_s = min(self.retry_max_seconds, self.retry_base_seconds * (2 ** (attempt - 1)))
                sleep_s *= (0.7 + random.random() * 0.6)
                logger.warning(
                    "OCR transport error attempt=%s/%s err=%r sleeping=%.2fs",
                    attempt, self.max_retries, e, sleep_s
                )
                await asyncio.sleep(sleep_s)

        if last_resp is not None:
            return last_resp
        raise RuntimeError(f"OCR HTTP error tras retries: {last_exc}")

    def count_pages(self, pdf_bytes: bytes) -> int:
        if not pdf_bytes:
            return 0
        reader = PdfReader(BytesIO(pdf_bytes))
        return len(reader.pages)

    def _payload_for_url(self, pdf_url: str) -> dict:
        return {
            "model": self.model,
            "document": {"type": "document_url", "document_url": pdf_url},
            "include_image_base64": self.include_image_base64,
        }

    def _payload_for_bytes(self, pdf_bytes: bytes) -> dict:
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        data_uri = f"data:application/pdf;base64,{b64}"
        return {
            "model": self.model,
            "document": {"type": "document_url", "document_url": data_uri},
            "include_image_base64": self.include_image_base64,
        }

    async def pdf_url_to_text(self, pdf_url: str) -> str:
        if not pdf_url:
            raise ValueError("pdf_url vacío")

        resp = await self._post_with_retries(self._payload_for_url(pdf_url))
        if resp.status_code >= 400:
            raise RuntimeError(f"OCR falló ({resp.status_code}): {resp.text}")

        data = resp.json()
        text = self._extract_text_defensive(data)
        if not text:
            raise RuntimeError("OCR respondió, pero no se pudo extraer texto del JSON")
        return text

    async def pdf_bytes_to_text(self, pdf_bytes: bytes) -> str:
        """
        OCR por bytes. Si el PDF es grande, lo parte en chunks válidos (pypdf).
        """
        if not pdf_bytes:
            raise ValueError("pdf_bytes está vacío")

        total_pages = self.count_pages(pdf_bytes)
        size_mb = len(pdf_bytes) / (1024 * 1024)
        logger.info("OCR input bytes: size=%.2fMB pages=%s", size_mb, total_pages)

        # Si cabe razonablemente y está bajo max páginas, intentamos 1 solo request.
        if total_pages <= self.chunking.max_pages_per_request and len(pdf_bytes) <= self.chunking.target_chunk_max_bytes:
            return await self._ocr_single_bytes(pdf_bytes)

        # Si no, chunking adaptativo
        chunks = self._split_pdf_adaptive(pdf_bytes)

        texts: list[str] = []
        processed_pages = 0

        for idx, chunk in enumerate(chunks, start=1):
            chunk_pages = self.count_pages(chunk)
            start_page = processed_pages + 1
            end_page = processed_pages + chunk_pages

            part_text = await self._ocr_single_bytes(chunk, part_label=f"chunk#{idx} pages={chunk_pages} bytes={len(chunk)}")

            if self.add_part_headers:
                texts.append(f"--- PARTE {idx} (páginas {start_page}-{end_page}) ---\n{part_text}")
            else:
                texts.append(part_text)

            processed_pages += chunk_pages

        return "\n\n".join(texts).strip()

    async def pdf_to_text_prefer_url(self, *, pdf_url: Optional[str], pdf_bytes: Optional[bytes]) -> str:
        """
        Método recomendado:
        - intenta OCR por URL primero (barato y estable)
        - si falla por acceso/cliente, cae a bytes chunked.
        """
        if pdf_url:
            try:
                return await self.pdf_url_to_text(pdf_url)
            except Exception as e:
                # Si el URL no es accesible para el servicio (403/404), o hay problema puntual,
                # caemos a bytes si están disponibles.
                logger.warning("OCR por URL falló, fallback a bytes. err=%r", e)

        if pdf_bytes:
            return await self.pdf_bytes_to_text(pdf_bytes)

        raise ValueError("Debe proporcionar pdf_url o pdf_bytes")

    async def _ocr_single_bytes(self, pdf_bytes: bytes, part_label: str = "") -> str:
        """
        OCR de un PDF (chunk) en una sola llamada.
        """
        resp = await self._post_with_retries(self._payload_for_bytes(pdf_bytes))
        if resp.status_code >= 400:
            hint = f" ({part_label})" if part_label else ""
            raise RuntimeError(f"OCR falló{hint} ({resp.status_code}): {resp.text}")

        data = resp.json()
        text = self._extract_text_defensive(data)
        if not text:
            raise RuntimeError("OCR respondió, pero no se pudo extraer texto del JSON")
        return text

    def _split_pdf_adaptive(self, pdf_bytes: bytes) -> list[bytes]:
        """
        Divide el PDF en chunks válidos, respetando:
        - max_pages_per_request
        - target_chunk_max_bytes (aprox)
        Para estimar tamaño, vamos construyendo el writer incrementalmente.
        """
        reader = PdfReader(BytesIO(pdf_bytes))
        total_pages = len(reader.pages)

        max_pages = max(1, int(self.chunking.max_pages_per_request))
        target_max = max(256 * 1024, int(self.chunking.target_chunk_max_bytes))  # al menos 256KB
        min_pages = max(1, int(self.chunking.min_pages_per_chunk))

        chunks: list[bytes] = []
        i = 0

        while i < total_pages:
            writer = PdfWriter()
            start_i = i
            pages_added = 0
            last_good_bytes: Optional[bytes] = None
            last_good_pages = 0

            # Intentamos meter tantas páginas como sea posible sin pasarnos de tamaño ni max_pages
            while i < total_pages and pages_added < max_pages:
                writer.add_page(reader.pages[i])
                pages_added += 1

                buf = BytesIO()
                writer.write(buf)
                current = buf.getvalue()

                if len(current) <= target_max or pages_added <= min_pages:
                    last_good_bytes = current
                    last_good_pages = pages_added
                    i += 1
                    continue

                # Se pasó del tamaño: usamos el último bueno (si existe)
                break

            if last_good_bytes is None:
                # Caso extremo: ni siquiera 1 página produce un PDF bajo target_max.
                # Igual lo enviamos (no podemos partir más). Esto evita loop infinito.
                writer = PdfWriter()
                writer.add_page(reader.pages[start_i])
                buf = BytesIO()
                writer.write(buf)
                only_one = buf.getvalue()
                chunks.append(only_one)
                i = start_i + 1
                continue

            chunks.append(last_good_bytes)

            # Si nos pasamos y NO avanzamos i en el último intento, aseguramos avance
            if i == start_i:
                i += max(1, last_good_pages)

        # Telemetría útil
        total_mb = len(pdf_bytes) / (1024 * 1024)
        chunk_mbs = [len(c) / (1024 * 1024) for c in chunks]
        logger.info(
            "PDF chunking: total_pages=%s total=%.2fMB chunks=%s sizesMB=%s",
            total_pages, total_mb, len(chunks), [round(x, 2) for x in chunk_mbs]
        )

        return chunks