import os
import re
import json
import asyncio
from typing import Dict, Optional, List, AsyncGenerator
from fastapi import HTTPException
from openai import AsyncAzureOpenAI


class ExtractionAgent:
    """
    Agente de extracción post-OCR usando Azure OpenAI.
    - Todo async
    - Soporta texto OCR largo con ventanas heurísticas + chunking
    """

    # ----------------------------
    # Prompts (tuyos, compactados)
    # ----------------------------
    PROMPT_DEMANDADO = """You are extracting legal entities from Colombian court rulings.
The field is: Demandado.
Instructions:
- Look for the party that appears after the word "contra" or is clearly described as the "Demandado".
- If the document only mentions "recurrente", "interpuesto por", or "demandante", that is NOT the Demandado.
- Typical format: one or more names in uppercase.
- If you cannot find a clear demandado, respond exactly: "Not found in the document".
Return ONLY the value for Demandado (no extra words)."""

    PROMPT_RADICADO = """Extract Radicado from Colombian court rulings.
Rules:
- If text contains patterns like "Radicación n. 13001-31-03-004-2015-00218-01" output only: 13001-31-03-004-2015-00218-01
- Or "Ref: 11001-3103-010-2000-00155-01" output only: 11001-3103-010-2000-00155-01
- Or "Referencia: R - 11001-0203-000-2006-00545-00" output only: 11001-0203-000-2006-00545-00 (note: remove labels and spaces)
- If none found, output exactly: "Not found in the document"
Return ONLY the radicado value (or the exact Not found sentence)."""

    PROMPT_DEMANDANTE = """You are extracting legal entities from Colombian court rulings.
The field is: Demandante.
Instructions:
- Look for the party that solicits or files the demand in the "Antecedentes" section.
- Usually introduced with phrases like "El demandante solicitó", "interpuesto por", or "quien demanda".
- The demandante is the one taking legal action.
- Typical format: one or more names in uppercase.
- If you cannot find a clear demandante, respond exactly: "Not found in the document".
Return ONLY the value for Demandante (no extra words)."""

    PROMPT_TIPO_PROCESO = """You are extracting the Tipo De Proceso from Colombian court rulings.
Instructions:
- Look for mentions of the phrase "en el proceso" or "dentro del proceso".
- Standardize output according to mappings:
  - "en el proceso declarativo" -> Declarativo
  - "dentro del proceso verbal" -> Verbal
  - "en el proceso ejecutivo singular" -> Ejecutivo singular
  - "en el proceso ordinario" -> Ordinario
  - "en el proceso de impugnación" -> Impugnación
  - "en el proceso incoado" -> Incoado
  - "en ejercicio de la acción popular" -> Acción popular
- If no explicit match, extract exact phrase immediately after "en el proceso" (or "dentro del proceso") without inferring.
- If uncertain or not found, respond exactly: "Not found in the document".
Return ONLY the value (no extra words)."""

    PROMPT_FECHA_SENTENCIA = """You are extracting the Fecha De Sentencia from Colombian court rulings.
Instructions:
- Look for expressions such as "frente a la sentencia de...", "respecto de la sentencia de...", or "contra la sentencia..."
- Normalize to DD/MM/YYYY (numbers/words/mixed).
- If no clear date, respond exactly: "Not found in the document".
Return ONLY the date (or Not found)."""

    PROMPT_FECHA_RECIBIDO = """Fecha De Recibido rules:
- If Radicación is 68001-31-03-007-2018-00134-01 output "2018"
- If Radicación is 11001-0203-000-2006-00545-00 output "2006"
- If Radicación is 13001-31-03-004-2015-00218-01 output "2015"
- The order to extract the data would be the fifth position.
If you doubt you have the answer, answer "Not found in the document".
Return ONLY the value (or Not found)."""

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        deployment: str = "gpt-4.1-mini",
        default_max_output_tokens: int = 800,
    ):
        self.endpoint = "https://camil-mjxgel5c-eastus2.cognitiveservices.azure.com/"
        self.api_key = "56WV3DWq81XWh22VR1lecS5EtS9YP7MsWjJ9JtwGEBaulu3RDGvWJQQJ99CAACHYHv6XJ3w3AAAAACOGM4om"
        self.api_version = "2024-12-01-preview"
        self.deployment = "gpt-4.1-mini"
        self.default_max_output_tokens = default_max_output_tokens

        self._client: Optional[AsyncAzureOpenAI] = None

    # ----------------------------
    # Lifecycle
    # ----------------------------
    async def startup(self) -> None:
        if not self.endpoint or not self.api_key:
            # No rompas el server en startup; pero sí fallará en extract()
            self._client = None
            return

        self._client = AsyncAzureOpenAI(
            api_version=self.api_version,
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
        )

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ----------------------------
    # Public API
    # ----------------------------
    async def extract_all(self, ocr_text: str) -> Dict[str, str]:
        text = self._normalize_text(ocr_text)
        windows = self._extract_windows(text)

        radicado = await self._robust_extract(self.PROMPT_RADICADO, text, windows.get("radicado"))

        demandado_task = asyncio.create_task(self._robust_extract(self.PROMPT_DEMANDADO, text, windows.get("demandado")))
        demandante_task = asyncio.create_task(self._robust_extract(self.PROMPT_DEMANDANTE, text, windows.get("demandante")))
        tipo_proceso_task = asyncio.create_task(self._robust_extract(self.PROMPT_TIPO_PROCESO, text, windows.get("tipo_proceso")))
        fecha_sent_task = asyncio.create_task(self._robust_extract(self.PROMPT_FECHA_SENTENCIA, text, windows.get("fecha_sentencia")))

        demandado, demandante, tipo_proceso, fecha_sentencia = await asyncio.gather(
            demandado_task, demandante_task, tipo_proceso_task, fecha_sent_task
        )

        fecha_recibido_input = f"Radicado: {radicado}\n\nText:\n{windows.get('radicado') or text[:8000]}"
        fecha_de_recibido = await self._call_llm(self.PROMPT_FECHA_RECIBIDO, fecha_recibido_input, max_tokens=30)

        return {
            "demandado": demandado,
            "demandante": demandante,
            "radicado": radicado,
            "tipo_de_proceso": tipo_proceso,
            "fecha_de_sentencia": fecha_sentencia,
            "fecha_de_recibido": fecha_de_recibido or "Not found in the document",
        }

    async def stream_extract(self, ocr_text: str) -> AsyncGenerator[str, None]:
        """
        Genera eventos SSE (Server-Sent Events) en formato:
        data: {"field":"...", "value":"..."}\n\n
        """
        text = self._normalize_text(ocr_text)
        windows = self._extract_windows(text)

        radicado = await self._robust_extract(self.PROMPT_RADICADO, text, windows.get("radicado"))
        yield f"data: {json.dumps({'field': 'radicado', 'value': radicado})}\n\n"

        tasks = {
            "demandado": asyncio.create_task(self._robust_extract(self.PROMPT_DEMANDADO, text, windows.get("demandado"))),
            "demandante": asyncio.create_task(self._robust_extract(self.PROMPT_DEMANDANTE, text, windows.get("demandante"))),
            "tipo_de_proceso": asyncio.create_task(self._robust_extract(self.PROMPT_TIPO_PROCESO, text, windows.get("tipo_proceso"))),
            "fecha_de_sentencia": asyncio.create_task(self._robust_extract(self.PROMPT_FECHA_SENTENCIA, text, windows.get("fecha_sentencia"))),
        }

        for name, task in tasks.items():
            value = await task
            yield f"data: {json.dumps({'field': name, 'value': value})}\n\n"

        fecha_recibido_input = f"Radicado: {radicado}\n\nText:\n{windows.get('radicado') or text[:8000]}"
        fecha_de_recibido = await self._call_llm(self.PROMPT_FECHA_RECIBIDO, fecha_recibido_input, max_tokens=30)
        yield f"data: {json.dumps({'field': 'fecha_de_recibido', 'value': fecha_de_recibido})}\n\n"

        yield "data: {\"done\": true}\n\n"

    # ----------------------------
    # Internals
    # ----------------------------
    def _normalize_text(self, t: str) -> str:
        t = t.replace("\u00a0", " ")
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    def _split_into_chunks(self, text: str, chunk_size: int = 12000, overlap: int = 800) -> List[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - overlap)
        return chunks

    def _extract_windows(self, text: str) -> Dict[str, str]:
        lower = text.lower()

        def window_around(keyword: str, radius: int = 3500) -> Optional[str]:
            idx = lower.find(keyword)
            if idx == -1:
                return None
            s = max(0, idx - radius)
            e = min(len(text), idx + radius)
            return text[s:e]

        windows = {
            "demandado": window_around("contra"),
            "tipo_proceso": window_around("en el proceso") or window_around("dentro del proceso"),
            "fecha_sentencia": window_around("sentencia"),
            "radicado": window_around("radicación") or window_around("ref:") or window_around("referencia"),
            "demandante": window_around("antecedentes") or window_around("demandante") or window_around("interpuesto por"),
        }

        if not windows["radicado"]:
            windows["radicado"] = text[:8000]

        return {k: v for k, v in windows.items() if v}

    async def _call_llm(self, prompt: str, user_text: str, max_tokens: Optional[int] = None) -> str:
        if self._client is None:
            raise HTTPException(status_code=500, detail="Azure OpenAI client not configured (missing env vars).")

        resp = await self._client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_completion_tokens=max_tokens or self.default_max_output_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    async def _robust_extract(self, prompt: str, full_text: str, preferred_window: Optional[str] = None) -> str:
        if preferred_window:
            out = await self._call_llm(prompt, preferred_window)
            if out and out != "Not found in the document":
                return out

        for ch in self._split_into_chunks(full_text):
            out = await self._call_llm(prompt, ch)
            if out and out != "Not found in the document":
                return out

        return "Not found in the document"