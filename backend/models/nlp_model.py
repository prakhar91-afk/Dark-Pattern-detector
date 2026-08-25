"""
models/nlp_model.py — NLP dark-pattern detection pipeline
==========================================================
Uses a fine-tuned DistilBERT (or RoBERTa) model for multi-label
classification of dark-pattern text. Falls back to zero-shot
classification via facebook/bart-large-mnli if no checkpoint exists.
"""

from __future__ import annotations
import os
import re
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from loguru import logger

# ─── Constants ────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "nlp"
FINE_TUNED_MODEL = CHECKPOINT_DIR / "best_model"

# Category labels → human descriptions
CATEGORIES = {
    "urgency":         "Fake urgency / countdown pressure",
    "optin":           "Pre-checked opt-in / forced consent",
    "confirm_shaming": "Confirm-shaming language",
    "hidden_flows":    "Hidden unsubscribe or cancel flow",
    "disguised_ads":   "Disguised advertisement",
}

# Zero-shot hypothesis templates
ZS_HYPOTHESES = {
    "urgency":         "This text creates artificial urgency or time pressure to manipulate users.",
    "optin":           "This text tricks users into subscribing or consenting without realising it.",
    "confirm_shaming": "This text shames or guilts users for declining an offer.",
    "hidden_flows":    "This text conceals or obstructs the ability to unsubscribe or cancel.",
    "disguised_ads":   "This text disguises an advertisement as organic content.",
}

# Rule-based regex patterns for fast pre-filtering (reduces API/model calls)
RULE_PATTERNS: Dict[str, List[re.Pattern]] = {
    "urgency": [
        re.compile(r'\b(only\s+\d+\s+left|limited\s+time|hurry|ends?\s+in|expires?\s+in|today\s+only|last\s+chance|don\'?t\s+miss)\b', re.I),
        re.compile(r'\b(flash\s+sale|deal\s+expires|offer\s+ends|selling\s+fast|almost\s+gone)\b', re.I),
        re.compile(r'\d+\s*(hour|minute|second|hr|min|sec)s?\s*(left|remaining|only)', re.I),
    ],
    "optin": [
        re.compile(r'\b(yes[,!]?\s+sign\s+me\s+up|keep\s+me\s+informed|subscribe\s+to\s+receive)\b', re.I),
        re.compile(r'\b(i\s+agree\s+to\s+receive|send\s+me\s+(offers|deals|news))\b', re.I),
    ],
    "confirm_shaming": [
        re.compile(r'\bno\s*thanks[,.]?\s*i\s*(hate|don\'?t|am\s+not)\b', re.I),
        re.compile(r'\bno[,.]?\s*i\s*don\'?t\s*want\s+to\s+(save|improve|get|learn)\b', re.I),
        re.compile(r'\bdecline\s+(this\s+)?(amazing|great|exclusive|free|special)\b', re.I),
        re.compile(r'\bi\'?m\s+(fine|ok|okay)\s+paying\s+(more|full\s+price)\b', re.I),
    ],
    "hidden_flows": [
        re.compile(r'\b(unsubscribe|opt.?out|cancel\s+(anytime|subscription|membership))\b', re.I),
    ],
    "disguised_ads": [
        re.compile(r'\b(sponsored|promoted|partnered\s+content|native\s+ad|in\s+association\s+with)\b', re.I),
    ],
}

CONFIDENCE_BOOST_RULES = 0.85    # confidence for rule matches
CONFIDENCE_ZERO_SHOT_THRESHOLD = 0.55  # minimum ZS score to report


class NLPModel:
    """
    Two-stage dark-pattern text classifier:
      1. Rule-based regex pre-filter (fast, deterministic)
      2. Transformer model inference (accurate, slower)
         - Fine-tuned DistilBERT if checkpoint exists
         - Zero-shot BART-MNLI as universal fallback
    """

    def __init__(self):
        self._pipeline = None
        self._pipeline_type: str = "none"
        self._load_model()

    def _load_model(self):
        """Load the best available model."""
        if FINE_TUNED_MODEL.exists():
            self._load_fine_tuned()
        else:
            logger.info("[NLP] No fine-tuned checkpoint found — using zero-shot fallback")
            self._load_zero_shot()

    def _load_fine_tuned(self):
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "text-classification",
                model=str(FINE_TUNED_MODEL),
                tokenizer=str(FINE_TUNED_MODEL),
                device=-1,   # CPU; set to 0 for GPU
                top_k=None,  # return all labels
                truncation=True,
                max_length=128,
            )
            self._pipeline_type = "fine_tuned"
            logger.success(f"[NLP] Fine-tuned model loaded from {FINE_TUNED_MODEL}")
        except Exception as e:
            logger.warning(f"[NLP] Fine-tuned load failed ({e}), falling back to zero-shot")
            self._load_zero_shot()

    def _load_zero_shot(self):
        try:
            from transformers import pipeline as hf_pipeline
            self._pipeline = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,
            )
            self._pipeline_type = "zero_shot"
            logger.success("[NLP] Zero-shot BART-MNLI loaded")
        except Exception as e:
            logger.error(f"[NLP] Could not load any transformer model: {e}")
            self._pipeline = None
            self._pipeline_type = "rules_only"

    # ─── Public API ───────────────────────────────────────────────
    def analyze(self, texts: List[Dict]) -> List[Dict]:
        """
        Analyze a list of text dicts (from content.js DOM extraction).
        Returns detections list in backend Detection format.
        """
        if not texts:
            return []

        t0 = time.time()
        detections = []

        for item in texts:
            text     = item.get("text", "").strip()
            selector = item.get("selector", "")
            rect     = item.get("rect", {})

            if not text or len(text) < 5:
                continue

            # Stage 1: rule-based
            rule_hits = self._rule_scan(text)
            for category, confidence in rule_hits:
                detections.append({
                    "type":        category,
                    "severity":    self._severity_from_confidence(confidence),
                    "confidence":  confidence,
                    "selector":    selector,
                    "rect":        rect,
                    "description": f"[NLP/Rule] {CATEGORIES[category]}",
                    "source":      "nlp",
                    "text":        text[:120],
                })

            # Stage 2: model inference (only if no rule hit)
            if not rule_hits and self._pipeline:
                model_hits = self._model_scan(text)
                for category, confidence in model_hits:
                    detections.append({
                        "type":        category,
                        "severity":    self._severity_from_confidence(confidence),
                        "confidence":  confidence,
                        "selector":    selector,
                        "rect":        rect,
                        "description": f"[NLP/Model] {CATEGORIES[category]}",
                        "source":      "nlp",
                        "text":        text[:120],
                    })

        elapsed = int((time.time() - t0) * 1000)
        logger.debug(f"[NLP] Analyzed {len(texts)} texts → {len(detections)} detections in {elapsed}ms")
        return detections

    @property
    def model_type(self) -> str:
        return self._pipeline_type

    # ─── Stage 1: rule scan ───────────────────────────────────────
    def _rule_scan(self, text: str) -> List[Tuple[str, float]]:
        hits = []
        for category, patterns in RULE_PATTERNS.items():
            for pattern in patterns:
                if pattern.search(text):
                    hits.append((category, CONFIDENCE_BOOST_RULES))
                    break
        return hits

    # ─── Stage 2: model inference ─────────────────────────────────
    def _model_scan(self, text: str) -> List[Tuple[str, float]]:
        if self._pipeline_type == "fine_tuned":
            return self._fine_tuned_scan(text)
        elif self._pipeline_type == "zero_shot":
            return self._zero_shot_scan(text)
        return []

    def _fine_tuned_scan(self, text: str) -> List[Tuple[str, float]]:
        """Multi-label classification from fine-tuned model."""
        try:
            results = self._pipeline(text[:512])
            hits = []
            for label_score in (results[0] if isinstance(results[0], list) else results):
                label = label_score["label"].lower().replace(" ", "_").replace("-", "_")
                score = label_score["score"]
                if label in CATEGORIES and score >= CONFIDENCE_ZERO_SHOT_THRESHOLD:
                    hits.append((label, round(score, 3)))
            return hits
        except Exception as e:
            logger.warning(f"[NLP] Fine-tuned inference error: {e}")
            return []

    def _zero_shot_scan(self, text: str) -> List[Tuple[str, float]]:
        """Zero-shot classification against each category hypothesis."""
        try:
            hypotheses = list(ZS_HYPOTHESES.values())
            result = self._pipeline(
                text[:512],
                candidate_labels=hypotheses,
                multi_label=True,
            )
            hits = []
            cat_keys = list(ZS_HYPOTHESES.keys())
            for label, score in zip(result["labels"], result["scores"]):
                # Map hypothesis back to category key
                try:
                    idx = hypotheses.index(label)
                    category = cat_keys[idx]
                    if score >= CONFIDENCE_ZERO_SHOT_THRESHOLD:
                        hits.append((category, round(score, 3)))
                except ValueError:
                    pass
            return hits
        except Exception as e:
            logger.warning(f"[NLP] Zero-shot inference error: {e}")
            return []

    @staticmethod
    def _severity_from_confidence(confidence: float) -> str:
        if confidence >= 0.80: return "high"
        if confidence >= 0.60: return "medium"
        return "low"
