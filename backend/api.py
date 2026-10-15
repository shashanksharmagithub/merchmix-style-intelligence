from __future__ import annotations

import json
import mimetypes
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from backend.model_service import (
    PredictionDataUnavailable,
    SeasonalDataUnavailable,
    get_project_status,
    get_seasonal_styles,
    get_seasons,
    get_style,
    get_top_styles,
    refresh_cache,
)
from data_science.config import CONFIG, PROJECT_ROOT

try:
    from fastapi import FastAPI, HTTPException
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import RedirectResponse
    from starlette.staticfiles import StaticFiles

    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:
    FASTAPI_AVAILABLE = False


if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Next-Season Style Intelligence API",
        description=(
            "Read-only business API for ranked style forecasts, model-based explanation signals, "
            "historical seasonal winners, and project health. Prediction artifacts are created by "
            "the offline data-science pipeline."
        ),
        version="1.2.0",
        openapi_tags=[
            {"name": "status", "description": "Artifact and service readiness."},
            {"name": "forecast", "description": "Forecasted winning styles and detail records."},
            {"name": "seasonal", "description": "Descriptive historical season views."},
            {"name": "administration", "description": "Local artifact-cache controls."},
        ],
        swagger_ui_parameters={"defaultModelsExpandDepth": 1, "displayRequestDuration": True},
    )

    @app.get(
        "/health",
        tags=["status"],
        summary="Check service and artifact readiness",
        response_description="Service status plus availability of model, prediction, concept, and seasonal artifacts.",
    )
    def health() -> dict:
        return {"status": "ok", **get_project_status()}

    @app.get(
        "/styles/top",
        tags=["forecast"],
        summary="List the top predicted styles",
        description=(
            "Returns the rolling-backtest-selected forecast winners with normalized ranking scores, "
            "predicted units, sales history, product attributes, and model-based explanation metadata."
        ),
        responses={503: {"description": "Prediction artifacts have not been generated."}},
    )
    def styles_top() -> dict:
        try:
            return get_top_styles()
        except PredictionDataUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(
        "/styles/season/{season}",
        tags=["seasonal"],
        summary="List historical winners for one season",
        description=(
            "Returns descriptive historical winners for spring, summer, autumn, or winter. "
            "This endpoint is not a separately trained seasonal forecast."
        ),
        responses={
            404: {"description": "The season is not one of spring, summer, autumn, or winter."},
            503: {"description": "Seasonal artifacts have not been generated."},
        },
    )
    def styles_by_season(season: str) -> dict:
        try:
            payload = get_seasonal_styles(season)
        except SeasonalDataUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if payload is None:
            raise HTTPException(status_code=404, detail=f"Unknown season: {season}")
        return payload

    @app.get(
        "/seasons",
        tags=["seasonal"],
        summary="List all historical seasonal views",
        responses={503: {"description": "Seasonal artifacts have not been generated."}},
    )
    def seasons() -> dict:
        try:
            return get_seasons()
        except SeasonalDataUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get(
        "/styles/{style_id}",
        tags=["forecast"],
        summary="Get one forecast-winner detail record",
        description=(
            "Looks up a style by stable product-code style_id. Detail includes model score, "
            "grouped counterfactual explanation signals, product attributes, and historical performance."
        ),
        responses={
            404: {"description": "The style_id is not present in the current top-style artifact."},
            503: {"description": "Prediction artifacts have not been generated."},
        },
    )
    def style_detail(style_id: str) -> dict:
        try:
            style = get_style(style_id)
        except PredictionDataUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if style is None:
            raise HTTPException(status_code=404, detail=f"Unknown style_id: {style_id}")
        return style

    @app.post(
        "/admin/refresh",
        tags=["administration"],
        summary="Reload prediction artifacts on the next request",
        description="Clears in-process JSON caches after pipeline artifacts are regenerated locally.",
    )
    def admin_refresh() -> dict:
        refresh_cache()
        return {"status": "refreshed"}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", include_in_schema=False)
    def frontend_redirect() -> RedirectResponse:
        return RedirectResponse(url="/app/")

    image_dir = CONFIG.images_dir if CONFIG.images_dir.exists() else CONFIG.deployment_images_dir
    if image_dir.exists():
        app.mount("/images", StaticFiles(directory=str(image_dir)), name="images")

    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.exists():
        app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    if CONFIG.concept_image_path.exists():
        app.mount("/outputs", StaticFiles(directory=str(CONFIG.concept_image_path.parent)), name="outputs")

else:
    app = None

    class StdlibAPIHandler(SimpleHTTPRequestHandler):
        """Small dependency-free server for local walkthroughs without FastAPI installed."""

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, root: Path, relative: str) -> None:
            target = (root / relative).resolve()
            if not str(target).startswith(str(root.resolve())) or not target.exists() or target.is_dir():
                self._send_json({"detail": "Not found"}, 404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def do_POST(self) -> None:
            if self.path.rstrip("/") == "/admin/refresh":
                refresh_cache()
                self._send_json({"status": "refreshed"})
                return
            self._send_json({"detail": "Not found"}, 404)

        def do_GET(self) -> None:
            path = unquote(self.path.split("?", 1)[0])
            if path == "/health":
                self._send_json({"status": "ok", **get_project_status()})
                return
            if path == "/styles/top":
                try:
                    self._send_json(get_top_styles())
                except PredictionDataUnavailable as exc:
                    self._send_json({"detail": str(exc), "status": get_project_status()}, 503)
                return
            if path == "/seasons":
                try:
                    self._send_json(get_seasons())
                except SeasonalDataUnavailable as exc:
                    self._send_json({"detail": str(exc), "status": get_project_status()}, 503)
                return
            if path.startswith("/styles/season/"):
                season = path.removeprefix("/styles/season/")
                try:
                    payload = get_seasonal_styles(season)
                except SeasonalDataUnavailable as exc:
                    self._send_json({"detail": str(exc), "status": get_project_status()}, 503)
                    return
                if payload is None:
                    self._send_json({"detail": f"Unknown season: {season}"}, 404)
                    return
                self._send_json(payload)
                return
            if path.startswith("/styles/"):
                style_id = path.removeprefix("/styles/")
                try:
                    style = get_style(style_id)
                except PredictionDataUnavailable as exc:
                    self._send_json({"detail": str(exc), "status": get_project_status()}, 503)
                    return
                if style is None:
                    self._send_json({"detail": f"Unknown style_id: {style_id}"}, 404)
                    return
                self._send_json(style)
                return
            if path in {"/app", "/app/"}:
                self._send_file(PROJECT_ROOT / "frontend", "index.html")
                return
            if path.startswith("/app/"):
                self._send_file(PROJECT_ROOT / "frontend", path.removeprefix("/app/"))
                return
            if path.startswith("/images/") and CONFIG.images_dir.exists():
                self._send_file(CONFIG.images_dir, path.removeprefix("/images/"))
                return
            if path.startswith("/outputs/"):
                self._send_file(CONFIG.concept_image_path.parent, path.removeprefix("/outputs/"))
                return
            self._send_json({"detail": "Not found"}, 404)


if __name__ == "__main__":
    if FASTAPI_AVAILABLE:
        import uvicorn

        uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=True)
    else:
        server = ThreadingHTTPServer(("127.0.0.1", 8000), StdlibAPIHandler)
        print("Serving dependency-free API at http://127.0.0.1:8000")
        print("Install requirements.txt to use the FastAPI/uvicorn server.")
        server.serve_forever()
