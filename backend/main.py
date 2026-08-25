"""
main.py — FastAPI backend for Dark Pattern Detector
====================================================
Endpoints:
  POST /analyze      — Main scan endpoint (DOM text + screenshot → trust score)
  GET  /health       — Backend + model status
  GET  /stats        — Aggregate scan statistics
  GET  /             — API info
"""

from __future__ import annotations
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from models.nlp_model import NLPModel
from models.cv_model import CVModel
from models.score_fusion import ScoreFusion, Detection


# ─── Global model instances ──────────────────────────────────────────────────
nlp_model:    Optional[NLPModel]    = None
cv_model:     Optional[CVModel]     = None
score_fusion: Optional[ScoreFusion] = None

# ─── Stats tracking ───────────────────────────────────────────────────────────
stats = {
    "total_scans": 0,
    "total_detections": 0,
    "avg_trust_score": 0.0,
    "avg_processing_ms": 0.0,
    "score_history": [],  # last 100 scores
}


# ─── Lifespan: load models on startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global nlp_model, cv_model, score_fusion

    logger.info("🚀 Starting Dark Pattern Detector backend...")
    logger.info("Loading NLP model...")
    nlp_model = NLPModel()

    logger.info("Loading CV model...")
    cv_model = CVModel()

    score_fusion = ScoreFusion()
    logger.success("✅ All models loaded — backend ready")

    yield

    logger.info("Shutting down backend...")


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Dark Pattern Detector API",
    description=(
        "Real-time detection of manipulative UI/UX dark patterns using "
        "NLP (DistilBERT/BART) and computer vision (YOLOv8)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Chrome extension origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1:8000",
        "chrome-extension://*",
        "*",  # Allow all for local dev; restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───────────────────────────────────────────────
class DOMTextItem(BaseModel):
    text:     str
    selector: Optional[str] = None
    rect:     Optional[Dict[str, int]] = None
    tag:      Optional[str] = None
    classes:  Optional[str] = None


class DOMElementItem(BaseModel):
    type:        str
    description: str
    selector:    Optional[str] = None
    rect:        Optional[Dict[str, int]] = None
    text:        Optional[str] = None


class AnalyzeRequest(BaseModel):
    url:            str           = Field(..., description="Page URL being scanned")
    dom_texts:      List[DOMTextItem]    = Field(default_factory=list)
    dom_elements:   List[DOMElementItem] = Field(default_factory=list)
    screenshot_b64: Optional[str] = Field(None, description="Base64 JPEG/PNG screenshot")
    page_title:     Optional[str] = None

    @field_validator("url")
    @classmethod
    def url_must_be_http(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v

    @field_validator("dom_texts")
    @classmethod
    def cap_dom_texts(cls, v):
        return v[:300]  # safety cap


class AnalyzeResponse(BaseModel):
    trust_score:          int
    grade:                str
    detections:           List[Dict[str, Any]]
    category_scores:      Dict[str, float]
    nlp_detection_count:  int
    cv_detection_count:   int
    rule_detection_count: int
    processing_ms:        int
    url:                  str
    model_info:           Dict[str, str]


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Dark Pattern Detector API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": {
            "nlp": nlp_model.model_type if nlp_model else "not_loaded",
            "cv":  cv_model.model_type  if cv_model  else "not_loaded",
        },
        "stats": {
            "total_scans":       stats["total_scans"],
            "total_detections":  stats["total_detections"],
            "avg_trust_score":   round(stats["avg_trust_score"], 1),
            "avg_processing_ms": round(stats["avg_processing_ms"], 1),
        }
    }


@app.get("/stats")
async def get_stats():
    return stats


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Main scan endpoint.
    Accepts DOM text + screenshot, returns trust score + detections.
    """
    t_start = time.time()

    if not nlp_model or not cv_model or not score_fusion:
        raise HTTPException(status_code=503, detail="Models not yet loaded")

    logger.info(f"Analyzing: {request.url[:80]}")

    # ── 1. Convert rule-based structural elements from content.js ────────────
    rule_detections = []
    for elem in request.dom_elements:
        cat = elem.type if elem.type in [
            "urgency", "optin", "confirm_shaming", "hidden_flows", "disguised_ads"
        ] else "general"
        rule_detections.append(Detection(
            type        = cat,
            severity    = "high" if cat in ("urgency", "confirm_shaming", "optin") else "medium",
            confidence  = 0.88,   # rule-based = high deterministic confidence
            selector    = elem.selector,
            rect        = elem.rect,
            description = elem.description,
            source      = "rule",
            text        = elem.text,
        ))

    # ── 2. NLP analysis on DOM texts ─────────────────────────────────────────
    text_dicts = [item.model_dump() for item in request.dom_texts]
    nlp_raw = nlp_model.analyze(text_dicts)
    nlp_detections = [
        Detection(
            type        = d["type"],
            severity    = d["severity"],
            confidence  = d["confidence"],
            selector    = d.get("selector"),
            rect        = d.get("rect"),
            description = d["description"],
            source      = "nlp",
            text        = d.get("text"),
        )
        for d in nlp_raw
    ]

    # ── 3. CV analysis on screenshot ─────────────────────────────────────────
    cv_raw = cv_model.analyze(request.screenshot_b64)
    cv_detections = [
        Detection(
            type        = d["type"],
            severity    = d["severity"],
            confidence  = d["confidence"],
            selector    = d.get("selector"),
            rect        = d.get("rect"),
            description = d["description"],
            source      = "cv",
            text        = None,
        )
        for d in cv_raw
    ]

    # ── 4. Score fusion ───────────────────────────────────────────────────────
    processing_ms = int((time.time() - t_start) * 1000)
    fusion_result = score_fusion.fuse(
        rule_detections = rule_detections,
        nlp_detections  = nlp_detections,
        cv_detections   = cv_detections,
        processing_ms   = processing_ms,
    )

    # ── 5. Update global stats ────────────────────────────────────────────────
    _update_stats(fusion_result.trust_score, len(fusion_result.detections), processing_ms)

    logger.info(
        f"✅ Scan complete | trust={fusion_result.trust_score} ({fusion_result.grade}) | "
        f"detections={len(fusion_result.detections)} | {processing_ms}ms"
    )

    return AnalyzeResponse(
        **fusion_result.to_dict(),
        url        = request.url,
        model_info = {
            "nlp": nlp_model.model_type,
            "cv":  cv_model.model_type,
        }
    )


# ─── Error handlers ───────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": type(exc).__name__}
    )


# ─── Stats helper ─────────────────────────────────────────────────────────────
def _update_stats(trust_score: int, detection_count: int, processing_ms: int):
    n = stats["total_scans"]
    stats["total_scans"]      += 1
    stats["total_detections"] += detection_count
    stats["avg_trust_score"]   = (stats["avg_trust_score"] * n + trust_score) / (n + 1)
    stats["avg_processing_ms"] = (stats["avg_processing_ms"] * n + processing_ms) / (n + 1)

    history = stats["score_history"]
    history.append(trust_score)
    if len(history) > 100:
        history.pop(0)


# ─── Dev entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
