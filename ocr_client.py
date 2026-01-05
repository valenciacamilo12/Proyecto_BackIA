import os
import base64
from io import BytesIO
import httpx
from pypdf import PdfReader, PdfWriter


class AzureMistralOcrClient:
    """
    Cliente async para Azure Mistral Document AI OCR.

    - Recibe bytes de un PDF.
    - Si el PDF tiene más de MAX_PAGES_PER_REQUEST páginas, lo divide en chunks.
    - Envía cada chunk al OCR de forma secuencial (seguro para cuotas / throttling).
    - Retorna texto plano concatenado.
    """

    MAX_PAGES_PER_REQUEST = 30

    def __init__(
        self,
        api_key: str | None = None,
        ocr_url: str = "https://servicios.services.ai.azure.com/providers/mistral/azure/ocr",
        model: str = "mistral-document-ai-2505",
        timeout_seconds: float = 180.0,
        connect_timeout_seconds: float = 30.0,
        include_image_base64: bool = False,
        add_part_headers: bool = True,
    ) -> None:
        self.api_key = "3YAmxUlV0gx3xpmx8g3G9czkS3lXoWhbmMz65ZzDU1tHtpE7blwyJQQJ99CAACZoyfiXJ3w3AAAAACOGR08k"
        self.ocr_url = ocr_url
        self.model = model
        self.timeout = httpx.Timeout(timeout_seconds, connect=connect_timeout_seconds)
        self.include_image_base64 = include_image_base64
        self.add_part_headers = add_part_headers

    def _headers(self) -> dict:
        if not self.api_key:
            raise ValueError("Falta la API Key. Defina AZURE_API_KEY en variables de entorno.")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _extract_text_defensive(data: dict) -> str:
        """
        Extracción tolerante: intenta campos directos y luego recorre JSON.
        Ajusta esta función cuando tengas un ejemplo real del JSON del OCR.
        """
        # Campos directos comunes
        for key in ("text", "output_text", "content", "result"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        texts: list[str] = []

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    # claves típicas en OCR
                    if k in ("text", "content", "markdown") and isinstance(v, str) and v.strip():
                        texts.append(v.strip())
                    walk(v)
            elif isinstance(obj, list):
                for it in obj:
                    walk(it)

        walk(data)

        # Deduplicar manteniendo orden
        seen = set()
        uniq: list[str] = []
        for t in texts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)

        return "\n".join(uniq).strip()

    def _get_total_pages(self, pdf_bytes: bytes) -> int:
        reader = PdfReader(BytesIO(pdf_bytes))
        return len(reader.pages)

    def _split_pdf_into_chunks(self, pdf_bytes: bytes) -> list[bytes]:
        """
        Divide un PDF en N chunks de máximo MAX_PAGES_PER_REQUEST páginas.
        """
        reader = PdfReader(BytesIO(pdf_bytes))
        total_pages = len(reader.pages)

        chunks: list[bytes] = []
        for start in range(0, total_pages, self.MAX_PAGES_PER_REQUEST):
            end = min(start + self.MAX_PAGES_PER_REQUEST, total_pages)
            writer = PdfWriter()

            for i in range(start, end):
                writer.add_page(reader.pages[i])

            buffer = BytesIO()
            writer.write(buffer)
            chunks.append(buffer.getvalue())

        return chunks

    async def _ocr_single_pdf(self, pdf_bytes: bytes) -> str:
        """
        Hace OCR de un PDF (<= 30 páginas) en una sola llamada.
        """
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        data_uri = f"data:application/pdf;base64,{b64}"

        payload = {
            "model": self.model,
            "document": {
                "type": "document_url",
                "document_url": data_uri,
            },
            "include_image_base64": self.include_image_base64,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self.ocr_url,
                headers=self._headers(),
                json=payload,
            )

        if resp.status_code >= 400:
            # Upstream error (incluye límite de páginas, formato, auth, etc.)
            raise RuntimeError(f"OCR falló ({resp.status_code}): {resp.text}")

        data = resp.json()
        text = self._extract_text_defensive(data)

        if not text:
            raise RuntimeError("OCR respondió, pero no se pudo extraer texto del JSON")

        return text

    async def pdf_bytes_to_text(self, pdf_bytes: bytes) -> str:
        """
        Punto de entrada principal:
        - Si <= 30 páginas: 1 request
        - Si > 30 páginas: split + múltiples requests + join
        """
        if not pdf_bytes:
            raise ValueError("pdf_bytes está vacío")

        total_pages = self._get_total_pages(pdf_bytes)

        # Caso simple
        if total_pages <= self.MAX_PAGES_PER_REQUEST:
            return await self._ocr_single_pdf(pdf_bytes)

        # Caso: PDF largo -> chunks
        chunks = self._split_pdf_into_chunks(pdf_bytes)

        texts: list[str] = []
        processed_pages = 0

        for idx, chunk in enumerate(chunks, start=1):
            # Calcula rango aproximado de páginas para encabezado
            start_page = processed_pages + 1
            end_page = min(processed_pages + self.MAX_PAGES_PER_REQUEST, total_pages)

            try:
                part_text = await self._ocr_single_pdf(chunk)
            except Exception as e:
                raise RuntimeError(
                    f"Error procesando bloque {idx} (páginas {start_page}-{end_page}): {e}"
                )

            if self.add_part_headers:
                texts.append(f"--- PARTE {idx} (páginas {start_page}-{end_page}) ---\n{part_text}")
            else:
                texts.append(part_text)

            processed_pages += self.MAX_PAGES_PER_REQUEST

        return "\n\n".join(texts).strip()
