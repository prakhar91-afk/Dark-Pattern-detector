"""
models/score_fusion.py — Trust score computation from NLP + CV outputs
======================================================================
Combines detections from both pipelines into a single trust score (0–100)
and a letter grade. Also produces per-category severity breakdowns.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math


# ─── Category configuration ─────────────────────────────────────────────────
CATEGORY_CONFIG = {
    "urgency":         {"weight": 1.2, "label": "Fake Urgency"},
    "optin":           {"weight": 1.0, "label": "Pre-checked Opt-in"},
    "confirm_shaming": {"weight": 1.3, "label": "Confirm-shaming"},
    "hidden_flows":    {"weight": 1.4, "label": "Hidden Unsubscribe/Cancel"},
    "disguised_ads":   {"weight": 0.9, "label": "Disguised Ads"},
    "general":         {"weight": 0.8, "label": "Dark Pattern"},
}

# Severity → base penalty points
SEVERITY_PENALTIES = {
    "high":   28,
    "medium": 14,
    "low":    5,
}

# Confidence threshold below which we downgrade severity
CONFIDENCE_THRESHOLD = 0.55


@dataclass
class Detection:
    type: str                        # category key from CATEGORY_CONFIG
    severity: str                    # "high" | "medium" | "low"
    confidence: float                # 0.0 – 1.0
    selector: Optional[str] = None
    rect: Optional[Dict] = None
    description: str = ""
    source: str = "rule"             # "rule" | "nlp" | "cv"
    text: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "type":        self.type,
            "severity":    self.severity,
            "confidence":  round(self.confidence, 3),
            "selector":    self.selector,
            "rect":        self.rect,
            "description": self.description,
            "source":      self.source,
            "text":        self.text,
        }


@dataclass
class FusionResult:
    trust_score: int
    grade: str
    detections: List[Detection]
    category_scores: Dict[str, float]
    nlp_detection_count: int
    cv_detection_count: int
    rule_detection_count: int
    processing_ms: int = 0

    def to_dict(self) -> Dict:
        return {
            "trust_score":          self.trust_score,
            "grade":                self.grade,
            "detections":           [d.to_dict() for d in self.detections],
            "category_scores":      self.category_scores,
            "nlp_detection_count":  self.nlp_detection_count,
            "cv_detection_count":   self.cv_detection_count,
            "rule_detection_count": self.rule_detection_count,
            "processing_ms":        self.processing_ms,
        }


class ScoreFusion:
    """
    Combines rule-based, NLP, and CV detections into a unified trust score.

    Pipeline weights:
      - Rule-based detections:  applied at full weight (these are deterministic)
      - NLP detections:         60% contribution weight
      - CV detections:          40% contribution weight
    """

    NLP_WEIGHT  = 0.60
    CV_WEIGHT   = 0.40
    RULE_WEIGHT = 1.00

    def fuse(
        self,
        rule_detections: List[Detection],
        nlp_detections:  List[Detection],
        cv_detections:   List[Detection],
        processing_ms:   int = 0,
    ) -> FusionResult:
        """
        Merge all detections, compute penalties, and return a FusionResult.
        """

        # De-duplicate: if NLP and rule both flag the same selector/text, keep highest confidence
        all_detections = self._merge_detections(
            rule_detections, nlp_detections, cv_detections
        )

        # Compute per-category raw scores (sum of confidence for that category)
        category_raw: Dict[str, float] = {k: 0.0 for k in CATEGORY_CONFIG}
        for det in all_detections:
            cat = det.type if det.type in CATEGORY_CONFIG else "general"
            weight = self._source_weight(det.source)
            category_raw[cat] += det.confidence * weight

        # Normalise category scores to 0.0 – 1.0
        category_scores = {
            k: round(min(1.0, v / max(1, self._expected_max(k))), 3)
            for k, v in category_raw.items()
        }

        # Compute total penalty
        total_penalty = 0.0
        for det in all_detections:
            base    = SEVERITY_PENALTIES.get(det.severity, 5)
            conf    = max(0.0, min(1.0, det.confidence))
            cat_cfg = CATEGORY_CONFIG.get(det.type, CATEGORY_CONFIG["general"])
            src_wt  = self._source_weight(det.source)
            total_penalty += base * conf * cat_cfg["weight"] * src_wt

        # Apply diminishing returns so a single heavily-flagged page doesn't
        # always score 0 — use a log curve
        if total_penalty > 0:
            scaled_penalty = 60 * (1 - math.exp(-total_penalty / 80))
        else:
            scaled_penalty = 0.0

        trust_score = max(0, min(100, round(100 - scaled_penalty)))

        return FusionResult(
            trust_score          = trust_score,
            grade                = self._grade(trust_score),
            detections           = all_detections,
            category_scores      = category_scores,
            nlp_detection_count  = len(nlp_detections),
            cv_detection_count   = len(cv_detections),
            rule_detection_count = len(rule_detections),
            processing_ms        = processing_ms,
        )

    # ─── Private helpers ─────────────────────────────────────────
    def _source_weight(self, source: str) -> float:
        return {
            "rule": self.RULE_WEIGHT,
            "nlp":  self.NLP_WEIGHT,
            "cv":   self.CV_WEIGHT,
        }.get(source, 0.5)

    def _expected_max(self, category: str) -> float:
        """Normalisation denominator: max expected raw score per category."""
        return {
            "urgency":         2.0,
            "optin":           1.5,
            "confirm_shaming": 1.0,
            "hidden_flows":    1.5,
            "disguised_ads":   2.5,
            "general":         3.0,
        }.get(category, 2.0)

    def _merge_detections(
        self,
        rule_dets: List[Detection],
        nlp_dets:  List[Detection],
        cv_dets:   List[Detection],
    ) -> List[Detection]:
        """
        Merge detections. If two sources flag the same element (same selector),
        keep the one with the highest confidence; bump severity if both agree.
        """
        merged: Dict[str, Detection] = {}

        for det in [*rule_dets, *nlp_dets, *cv_dets]:
            # Deduplication key
            key = det.selector or det.description or det.type
            if key not in merged:
                merged[key] = det
            else:
                existing = merged[key]
                # Promote confidence
                if det.confidence > existing.confidence:
                    merged[key] = det
                # If both sources agree → elevate severity
                if existing.source != det.source:
                    merged[key].severity = self._elevate_severity(merged[key].severity)

        result = list(merged.values())
        # Sort: high severity first, then by confidence descending
        severity_order = {"high": 0, "medium": 1, "low": 2}
        result.sort(key=lambda d: (severity_order.get(d.severity, 3), -d.confidence))
        return result

    @staticmethod
    def _elevate_severity(current: str) -> str:
        return {"low": "medium", "medium": "high", "high": "high"}.get(current, current)

    @staticmethod
    def _grade(score: int) -> str:
        if score >= 90: return "A"
        if score >= 75: return "B"
        if score >= 60: return "C"
        if score >= 40: return "D"
        return "F"
