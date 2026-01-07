import os
import logging
from typing import Optional, Dict, Any

import httpx

log = logging.getLogger("status-client")


class BackStatusClient:
    """
    Cliente para actualizar estado en el Back:
      PATCH {BACK_BASE_URL}/cargas/{id_carga}/status

    PROCESSED:
      {"status":"PROCESSED","comment":"...","extracted":{...}}

    ERROR:
      {"status":"ERROR","comment":"..."}
    """

    def __init__(self, base_url: Optional[str] = None, timeout_s: float = 15.0) -> None:
        self.base_url = (
            base_url
            or os.getenv("BACK_BASE_URL", "https://proyectoback-h6ajcba8cpewd5bc.brazilsouth-01.azurewebsites.net")
        ).strip().rstrip("/")
        if not self.base_url:
            raise ValueError("Falta BACK_BASE_URL (ej: https://proyectoback-xxx.azurewebsites.net)")
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def close(self) -> None:
        await self._client.aclose()

    async def update_status(
        self,
        *,
        id_carga: str,
        status: str,
        comment: str,
        extracted: Optional[Dict[str, Any]] = None,
    ) -> None:
        url = f"{self.base_url}/cargas/{id_carga}/status"
        payload: Dict[str, Any] = {"status": status, "comment": comment}

        # Solo enviar extracted cuando aplique
        if extracted is not None and status == "PROCESSED":
            payload["extracted"] = extracted

        resp = await self._client.patch(url, json=payload)
        if resp.status_code >= 400:
            log.error("Back status update failed. url=%s code=%s body=%s", url, resp.status_code, resp.text)
            resp.raise_for_status()