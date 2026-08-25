# Dark Pattern Detector

> **Real-time detection of manipulative UI/UX dark patterns** — A Chrome Extension + FastAPI backend that scans any webpage, flags deceptive elements, and displays a live trust score.

![Architecture](https://img.shields.io/badge/stack-Chrome%20MV3%20%2B%20FastAPI%20%2B%20DistilBERT%20%2B%20YOLOv8-6366f1?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Features

| Category | Detection Method |
|---|---|
| 🔴 **Fake Urgency** (countdown timers, scarcity banners) | DOM rules + NLP + YOLOv8 |
| 🟠 **Pre-checked Opt-ins** (forced email subscriptions) | DOM rules (checkbox detection) |
| 🟣 **Confirm-shaming** ("No thanks, I hate savings") | NLP (regex + transformer) |
| 🔵 **Hidden Unsubscribe Flows** (buried cancel links) | DOM rules (contrast / size) |
| 🟡 **Disguised Ads** (sponsored content styled as organic) | DOM rules + YOLOv8 |

---

## Project Structure

```
Dark Pattern Detector/
├── extension/               # Chrome Extension (Manifest V3)
│   ├── manifest.json
│   ├── background.js        # Service Worker — orchestration + API calls
│   ├── content.js           # DOM scanner + overlay renderer
│   └── popup/
│       ├── popup.html       # Premium dark-mode dashboard
│       ├── popup.css
│       └── popup.js
│
└── backend/                 # FastAPI service
    ├── main.py              # FastAPI app + /analyze endpoint
    ├── models/
    │   ├── nlp_model.py     # DistilBERT / BART zero-shot NLP pipeline
    │   ├── cv_model.py      # YOLOv8 visual detection pipeline
    │   └── score_fusion.py  # Weighted trust score combinator
    ├── training/
    │   ├── data_prep.py     # Dataset download + preprocessing
    │   ├── train_nlp.py     # DistilBERT fine-tuning script
    │   └── train_cv.py      # YOLOv8 fine-tuning script
    ├── requirements.txt
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Quick Start

### 1. Backend Setup

**Option A — Direct (recommended for development)**

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Option B — Docker**

```bash
cd backend
docker-compose up --build
```

Verify the backend is running:
```
http://localhost:8000/health
http://localhost:8000/docs      ← Swagger UI
```

### 2. Install Chrome Extension

1. Open Chrome → `chrome://extensions`
2. Enable **Developer Mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `extension/` folder
5. Pin the extension to your toolbar

### 3. Scan a Page

- Navigate to any webpage (e.g., a booking or e-commerce site)
- Click the **Dark Pattern Detector** icon
- Hit **Scan This Page** — the popup shows the trust score + flagged elements, and colored overlays appear directly on the page

---

## Model Training (Optional)

The backend works **immediately out-of-the-box** using:
- **NLP**: `facebook/bart-large-mnli` zero-shot classification (no training required)
- **CV**: `yolov8n.pt` COCO pretrained with proxy class mapping

To improve accuracy, fine-tune on the real dataset:

### Fine-tune NLP (DistilBERT)

```bash
cd backend

# Step 1: Download and prepare the ec-darkpattern dataset
python training/data_prep.py

# Step 2: Fine-tune DistilBERT (requires ~4GB RAM, ~15 min on CPU / ~2 min on GPU)
python training/train_nlp.py --model distilbert-base-uncased --epochs 3

# Optional: Use RoBERTa for higher accuracy (slower)
python training/train_nlp.py --model roberta-base --epochs 3 --batch-size 8
```

Fine-tuned model is saved to `backend/checkpoints/nlp/best_model/` and **automatically used** on next server start.

### Fine-tune YOLOv8 (Visual Detection)

```bash
cd backend

# Option 1: Download a Roboflow dataset
# Go to https://universe.roboflow.com/ → search "dark pattern UI elements"
# Export in YOLOv8 format to backend/data/cv_dataset/

# Option 2: Use synthetic demo dataset (auto-generated)
python training/train_cv.py --epochs 50 --imgsz 640

# GPU training (recommended)
python training/train_cv.py --epochs 100 --batch 16 --device 0
```

Fine-tuned weights saved to `backend/checkpoints/cv/best.pt`.

---

## API Reference

### `POST /analyze`

**Request body:**
```json
{
  "url": "https://example.com",
  "dom_texts": [
    { "text": "Only 2 left!", "selector": "#stock-warning", "rect": {"x":100,"y":200,"w":200,"h":40} }
  ],
  "dom_elements": [
    { "type": "urgency", "description": "Countdown timer detected", "selector": "#countdown" }
  ],
  "screenshot_b64": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "trust_score": 35,
  "grade": "F",
  "detections": [
    {
      "type": "urgency",
      "severity": "high",
      "confidence": 0.91,
      "selector": "#countdown",
      "description": "[Rule] Countdown timer detected",
      "source": "rule"
    }
  ],
  "category_scores": {
    "urgency": 0.85,
    "optin": 0.10,
    "confirm_shaming": 0.0,
    "hidden_flows": 0.22,
    "disguised_ads": 0.45
  },
  "nlp_detection_count": 3,
  "cv_detection_count": 1,
  "rule_detection_count": 2,
  "processing_ms": 312
}
```

### `GET /health`

Returns backend status and loaded model types.

### `GET /stats`

Returns aggregate scan statistics (total scans, average trust score, etc.).

---

## Trust Score Computation

```
base_score = 100
penalty    = Σ (severity_weight × confidence × category_weight × source_weight)

source weights: rule=1.0, nlp=0.6, cv=0.4
severity:       high=28pts, medium=14pts, low=5pts
category bonus: confirm_shaming×1.3, hidden_flows×1.4, urgency×1.2

trust_score = max(0, 100 - 60×(1 - e^(-penalty/80)))   ← diminishing returns
```

**Grade mapping:**

| Score | Grade | Meaning |
|---|---|---|
| 90–100 | A | Excellent — no dark patterns detected |
| 75–89  | B | Good — minor issues only |
| 60–74  | C | Moderate — some manipulative elements |
| 40–59  | D | Suspicious — significant dark patterns |
| 0–39   | F | Deceptive — heavy manipulation detected |

---

## Dark Pattern Categories

### 🔴 Fake Urgency
Countdown timers, "X left in stock", "Deal ends in Y hours" — designed to rush decisions.

**Detected by:** DOM attribute scanning (`[data-countdown]`), text pattern matching (`\d+:\d+`, "only X left"), NLP transformer, YOLOv8 (timer widgets).

### 🟠 Pre-checked Opt-ins
Checkboxes for newsletter/marketing consent that are checked by default.

**Detected by:** `input[type="checkbox"][checked]` + label text analysis for subscription keywords.

### 🟣 Confirm-shaming
Dismiss buttons written to guilt users: *"No thanks, I hate saving money"*.

**Detected by:** Regex patterns on button/link text, NLP zero-shot classification.

### 🔵 Hidden Unsubscribe / Cancel
Opt-out links made nearly invisible via tiny font, low contrast, or off-screen placement.

**Detected by:** CSS computed style analysis (font-size < 11px, opacity < 0.4, contrast ratio), text matching for "unsubscribe / opt out / cancel".

### 🟡 Disguised Advertisements
Native ads, sponsored content, or partner links styled to look like organic recommendations.

**Detected by:** `[data-ad]`, `[aria-label*="sponsor"]`, ARIA attributes, YOLOv8 ad container detection.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Extension** | Chrome Manifest V3, Content Scripts, Service Worker |
| **DOM Analysis** | Vanilla JavaScript (CSS selector + computed style inspection) |
| **NLP Model** | DistilBERT / RoBERTa (fine-tuned) or BART-MNLI (zero-shot) |
| **CV Model** | YOLOv8n / YOLOv8s (Ultralytics) |
| **Backend** | FastAPI + Uvicorn |
| **Dataset** | ec-darkpattern (yamanalab/ec-darkpattern on GitHub) |
| **Containerisation** | Docker + Docker Compose |

---

## Dataset Attribution

- **Text dataset**: [yamanalab/ec-darkpattern](https://github.com/yamanalab/ec-darkpattern) — Yada et al. (2022)  
  *"Dark patterns in e-commerce: a dataset and its baseline evaluations"*
- **Base NLP model**: [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) — Hugging Face
- **Base CV model**: [YOLOv8n](https://github.com/ultralytics/ultralytics) — Ultralytics

---

## License

MIT © 2024 Dark Pattern Detector Project
