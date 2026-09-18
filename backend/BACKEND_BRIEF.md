# Screening Backend — Build Brief

**Project:** AI-Based Fake Identity & Document Screening System
**This document:** the complete brief for the Python backend. Build against it directly.
**Rename to `CLAUDE.md` at repo root** if you want it loaded automatically every session.

**Companion document:** `FRONTEND_BRIEF.md`. The contracts in §4 below are the *same
contracts* as `src/types/screening.ts` on the frontend. They must stay in lockstep.
If you change a field name here, change it there in the same commit.

---

## 0. Read this first

You are building the analysis engine behind a border checkpoint. A wrong answer here
means a traveller is wrongly detained, or a forged passport walks through.

Four rules govern every design decision:

1. **The system recommends. The officer decides.** No endpoint auto-clears anything.
2. **Never guess.** If confidence is low, abstain and say so. A withheld answer is a
   valid output; a fabricated one is not.
3. **Absence of evidence is not evidence of absence.** A module that timed out, or a
   database that was unreachable, is reported as a coverage gap — never as "clean".
4. **Every adverse finding carries coordinates.** A signal the officer cannot verify
   against the physical document is worthless.

---

## 1. The OCR decision — read before writing any extraction code

This is the question that determines whether the system works. The answer is
**two different readers for two different jobs**, not one OCR engine for everything.

### 1.1 The split

| Zone | What it is | Reader | Why |
|---|---|---|---|
| **MRZ** | The OCR-B strip at the bottom | **Our own CRNN** (already built, see §1.3) | Fixed font, 37-symbol closed charset, fixed line width, and embedded check digits. A specialist beats every general engine here — and the check digits let us distinguish a *misread* from a *forgery*. No off-the-shelf model does that. |
| **VIZ** | Printed name, dates, authority, visa fields | **RapidOCR** (PP-OCRv5 models on ONNX Runtime) | Apache 2.0, ~50–80 MB, 0.5–1 s/page on CPU, no PaddlePaddle framework dependency. Runs fully offline at an edge checkpoint. |

### 1.2 Why RapidOCR and not the alternatives

RapidOCR takes PaddleOCR's trained detection and recognition models and runs them on
ONNX Runtime, dropping the PaddlePaddle framework entirely. For a border checkpoint that
must run offline on modest hardware, that is decisive.

| Candidate | Verdict | Reason |
|---|---|---|
| **RapidOCR (PP-OCRv5 ONNX)** | **Use this** | Apache 2.0, smallest footprint, CPU-fast, ONNX everywhere, fine-tunable via PaddleOCR then re-exported |
| PaddleOCR (native) | Fallback only | Same models, better structure tools (PP-StructureV3), but the Paddle framework dependency is heavy and awkward to containerise for edge nodes |
| Surya | **Do not use** | Code is Apache 2.0 but the **model weights are a modified AI Pubs Open RAIL-M licence — free only for research, personal use, and companies under $5M**. A government border deployment is not covered. Licensing risk you cannot accept on this project. |
| Tesseract | No | Weak on ID-card layouts and stylised document fonts. Fine for clean scanned books, wrong here. |
| EasyOCR | No | ~3× slower, ~500 MB footprint. Batch tool, not a real-time lane tool. |
| docTR | Acceptable alternative | Apache 2.0, clean API. Slightly heavier than RapidOCR; pick it only if RapidOCR's detector underperforms on your document set — measure before switching. |
| VLM readers (PaddleOCR-VL, dots.ocr, Qwen-VL, olmOCR) | **Not as primary** | See §1.4 — this is a safety rule, not a preference |

### 1.3 The MRZ reader already exists

The `mrz/` package is written and tested. Vendor it into `app/ocr/mrz/` unchanged:

```
mrz/spec.py     ICAO 9303 charset, 7-3-1 check digit, TD1/TD2/TD3/MRV parsing.
                Validates against the official ICAO reference example.
mrz/synth.py    Synthetic generator + print-scan degradation. 2000/2000 records
                pass their own check digits.
mrz/detect.py   Locates the MRZ band, deskews, splits into lines. ~39/40 on
                synthetic pages. Classical CV, no training needed.
mrz/model.py    CRNN + CTC, 8.2M params.
mrz/data.py     Dataset + width-bucketed collate.
mrz/train.py    Training loop with severity curriculum.
mrz/decode.py   Checksum-constrained beam search. The core of the module.
mrz/infer.py    MRZReader — image in, fields + signals out.
```

**Weights are not trained yet.** Train them (`python -m mrz.train --epochs 30
--batch 64 --steps 400`, GPU, ~15k steps) before integration testing. Until weights
exist, `MRZReader` must be behind an interface so the pipeline can run with a stub that
replays fixture MRZ strings.

**The checksum-constrained decoder is the project's differentiator.** Do not replace it
with greedy decoding for speed. Its contract:

- a hypothesis in the top-K beam whose check digits validate → `VERIFIED`
- **no** hypothesis validates → `CHECKSUM_UNRECOVERABLE`, emitted as a **high-severity
  tamper signal**, not an OCR error

Measured: with realistic 0/O and 1/I confusion, greedy is wrong on 35/40 samples and the
checksum-constrained beam recovers 60% of those, reporting the rest as unrecoverable
rather than silently wrong.

### 1.4 The VLM rule — non-negotiable

**A vision-language model must never be the authoritative reader of an identity field.**

VLMs generate plausible text. On a blurred date of birth a VLM will confidently produce
a well-formed date that was never on the document. In a book scan that is a typo. Here
it is a person detained on a hallucinated value, with no way for the officer to tell.

Pipeline OCR (detect → recognise → per-character confidence) fails *visibly*: confidence
drops, the checksum fails, the field is flagged. That failure mode is the requirement.

A VLM may be used in exactly one place: as an **optional secondary cross-check** whose
disagreement with the primary reader raises a `VIZ_READER_DISAGREEMENT` signal for human
review. It never overwrites a primary value, and the system must run correctly with it
disabled. Gate it behind `settings.enable_vlm_crosscheck`, default `false`.

### 1.5 Model registry — offline is mandatory

Checkpoints lose connectivity. Nothing may download a model at runtime.

```
models/
  mrz_crnn/1.3.0/model.onnx          + sha256
  rapidocr_det/v5/det.onnx           + sha256
  rapidocr_rec/v5/rec.onnx, dict.txt + sha256
  scrfd/10g/scrfd.onnx               + sha256
  arcface/buffalo_l/w600k_r50.onnx   + sha256
  tamper_unet/0.9.2/model.onnx       + sha256
  pad/minifas/1.0/model.onnx         + sha256
```

- `scripts/fetch_models.py` downloads once, verifies SHA-256, writes `manifest.json`.
- On startup the app **verifies every hash and refuses to boot on mismatch**. A silently
  swapped model in a border system is a security incident.
- `manifest.json` versions go into every audit record (§8.2).

---

## 2. Stack

| Concern | Choice | Note |
|---|---|---|
| API | **FastAPI** + uvicorn | native async, needed for the parallel waves |
| Async | **asyncio** | the DAG in §5 is asyncio-native |
| CPU-bound work | **ProcessPoolExecutor** | forensics releases no GIL; never run it on the event loop |
| Inference | **onnxruntime** | one shared session per model, `intra_op_num_threads` tuned |
| Vision | **opencv-python-headless**, **numpy**, **scipy**, **scikit-image** | |
| OCR (VIZ) | **rapidocr** + **onnxruntime** | §1.2 |
| Face | **insightface** (SCRFD + ArcFace), ONNX | |
| DB | **PostgreSQL 16** + **pgvector** | records and embeddings in one system — simplifies edge nodes |
| ORM | **SQLAlchemy 2.0** async + **Alembic** | |
| Objects | **MinIO** (S3 API), encrypted at rest | originals preserved for forensics |
| Validation | **Pydantic v2** | the contracts in §4 |
| Config | **pydantic-settings**, env-driven | |
| Tests | **pytest** + **pytest-asyncio** + **httpx** | |
| Lint | **ruff** + **mypy** strict | |

No Celery in v1. The DAG is in-process asyncio; a queue adds a failure mode without
solving a problem you have yet. Add it when you need multi-node backpressure.

---

## 3. Repository layout

```
screening-backend/
  app/
    main.py                  # FastAPI app, lifespan, model loading + hash check
    config.py
    contracts/               # Pydantic models — mirrors frontend types exactly
      session.py  signals.py  events.py
    pipeline/
      orchestrator.py        # THE DAG. gates, waves, timeouts, degraded mode
      gates.py               # quality gate, document classifier
      branches/
        ocr.py               # MRZ + VIZ, MRZ<->VIZ cross-check
        forensics.py         # metadata, ELA, noise, copy-move, font, heatmap
        face.py              # detect, PAD, embed, 1:1
        ovd.py               # video-sweep angular reflectance (optional)
        template.py          # one-class normality per country/version
      wave2/
        rules.py             # deterministic validation
        database.py          # watchlist, stolen/lost, 1:N gallery, graph
      crossdoc.py            # Stage 5.5
      correlate.py           # Stage 6 spatial clustering
      fusion.py              # Stage 7 risk, confidence, abstention
    ocr/
      mrz/                   # vendored, unchanged
      viz.py                 # RapidOCR wrapper + field classifier
    forensics/               # one module per technique, all pure functions
    faces/
    storage/
      db.py  models.py  repositories.py  objects.py
      audit.py               # hash chain
    api/
      screening.py           # POST /screening, POST /decision
      stream.py              # WS /ws/screening/{id}
      history.py  health.py
    workers/
      pool.py                # ProcessPoolExecutor lifecycle
  models/                    # weights, gitignored, fetched by script
  scripts/
    fetch_models.py  seed_watchlist.py  seed_gallery.py
  tests/
    fixtures/                # the 15 cases, shared with the frontend
  alembic/
  docker-compose.yml
```

---

## 4. Contracts

`app/contracts/` is generated-adjacent to the frontend's `screening.ts`. Same names,
same shapes, `snake_case` on the wire.

```python
# contracts/signals.py
class Region(BaseModel):
    x: float; y: float; w: float; h: float      # NORMALISED 0..1, never pixels
    document_id: str

class Signal(BaseModel):
    id: str
    code: str                                    # 'MRZ_CHECKDIGIT_DOB'
    module: Literal['ocr','validation','tamper','face','ovd','template',
                    'database','graph','crossdoc','system']
    severity: Literal['info','low','medium','high','critical']
    weight: float
    detail: str                                  # a finished human sentence
    region: Region | None = None
    heatmap_url: str | None = None
    convergence_group: str | None = None
```

**Coordinates are normalised 0..1 against the rectified document image.** The frontend
scales them to whatever canvas size it renders. Never emit pixels.

**`detail` is written by the backend, not the frontend.** It is the sentence the officer
reads: `DOB check digit expected 4, read 7`. Write it well.

The full `ScreeningSession`, `ScreenedDocument`, `MrzLine`, `FaceResult`, `GraphResult`
and the `ScreeningEvent` union mirror `FRONTEND_BRIEF.md` §3 field for field. Read that
file and port it.

### MRZ ribbon data

The frontend's signature UI element needs per-group checksum state. `mrz/spec.py`
already produces it — map `MRZResult.checks` into:

```python
class MrzGroup(BaseModel):
    name: str            # 'doc_number' | 'birth_date' | 'expiry_date' | 'composite'
    start: int           # char index, inclusive
    end: int             # exclusive, EXCLUDES the check digit
    check_digit_index: int
    valid: bool
    expected: str | None = None    # populated only when invalid
    read: str | None = None
```

Emit real indices from the format layout, not approximations. The UI underlines exactly
those characters.

---

## 5. The pipeline — this is the architecture

Not linear, not flat parallel. A DAG with **two blocking gates** and **two parallel
waves**, per document, then a cross-document join.

```
intake + hash + audit open
        │
   ┌────▼──────────────┐
   │ GATE 1: QUALITY   │  BLOCKING → fail = RECAPTURE, no score at all
   └────┬──────────────┘
   ┌────▼──────────────┐
   │ GATE 2: CLASSIFY  │  BLOCKING → selects config for everything downstream
   └────┬──────────────┘
        │
  ┌─────┼─────┬─────────┬─────────┬─────────┐     WAVE 1 (parallel, independent)
  ▼     ▼     ▼         ▼         ▼         ▼
 OCR  FORENSICS  FACE   OVD   TEMPLATE
  │     │         │      │      │
  ├─────┴─────────┴──────┴──────┤                 WAVE 2 (depends on wave 1)
  ▼                             ▼
 RULES                     DATABASE + GRAPH
  └─────────────┬───────────────┘
                ▼
        CROSS-DOCUMENT JOIN (Stage 5.5)
                ▼
        CORRELATION (Stage 6)
                ▼
        FUSION (Stage 7)
                ▼
        stream → officer decision → seal
```

### 5.1 Why the gates block

Every downstream module degrades on bad input, but each degrades **differently and
silently**. A blurred capture raises ELA noise (false tamper), lowers OCR confidence
(false checksum failure), and depresses face similarity (false mismatch). Fused, that is
a risk score near 70 on a genuine passport that was merely photographed badly. At
thousands of passengers a day this is the failure that gets the system switched off.

Classification blocks for a different reason: it selects the MRZ format, the field map,
the reference font metrics, the template anomaly model, and the expected hologram
location. Without it every downstream module is running generically.

### 5.2 The one rule that will bite you

```python
# forensics receives the ORIGINAL BYTES. Everything else gets the rectified image.
ocr_task       = branch_ocr(doc.rectified, ctx)
forensics_task = branch_forensics(doc.original_path, ctx)   # <-- not rectified
```

EXIF, JPEG quantization tables and compression history are destroyed the instant the
file is re-encoded. Rectifying before running forensics destroys the evidence the module
exists to find. This is the most common architectural mistake in systems of this kind.

### 5.3 The orchestrator

```python
BUDGET_MS = {'ocr': 700, 'forensics': 1200, 'face': 600,
             'ovd': 900, 'template': 500}

async def wave_1(doc, live_frame, sweep, ctx) -> dict:
    tasks = {
        'ocr':       asyncio.create_task(branch_ocr(doc.rectified, ctx)),
        'forensics': asyncio.create_task(branch_forensics(doc.original_path, ctx)),
        'face':      asyncio.create_task(branch_face(doc.rectified, live_frame, ctx)),
        'template':  asyncio.create_task(branch_template(doc.rectified, ctx)),
    }
    if sweep is not None:
        tasks['ovd'] = asyncio.create_task(branch_ovd(sweep, ctx))

    results = {}
    for name, task in tasks.items():
        try:
            results[name] = await asyncio.wait_for(task, BUDGET_MS[name] / 1000)
        except asyncio.TimeoutError:
            results[name] = Degraded(name, 'timeout')
        except Exception as exc:
            logger.exception('branch %s failed', name)
            results[name] = Degraded(name, str(exc))
    return results
```

**Failure isolation is the point.** A failed branch degrades the result; it never crashes
the session and it never silently passes:

```python
if isinstance(results['face'], Degraded):
    signals.append(Signal(code='FACE_UNAVAILABLE', severity='info', weight=0,
                          detail='Face module did not complete.'))
    confidence *= 0.6
    coverage_flags.append('no_biometric')
```

The session then reports *screened without biometric verification*. It does **not**
report a clean pass.

### 5.4 Emit events as they land

Every branch completion publishes to the session's event bus immediately. Do not
accumulate and send one blob — the officer sees extracted fields at ~0.7 s that way,
against ~1.6 s for the full analysis.

```python
async def run(session_id, inputs):
    async with audit.open(session_id, inputs) as audit_ctx:
        await bus.publish(session_id, ReceivedEvent(...))
        docs = []
        for doc_input in inputs.documents:
            q = await quality_gate(doc_input)
            await bus.publish(session_id, QualityEvent(...))
            if not q.ok:
                return await finish_recapture(session_id, q)
            cls = await classify(doc_input)
            await bus.publish(session_id, ClassifiedEvent(...))
            docs.append(...)
        # per-document pipelines run concurrently
        per_doc = await asyncio.gather(*(run_document(d) for d in docs))
        cross   = cross_document(per_doc)
        await bus.publish(session_id, CrossDocEvent(signals=cross))
        clusters = correlate(all_signals(per_doc, cross))
        verdict  = fuse(clusters, coverage_flags)
        await bus.publish(session_id, DecisionEvent(...))
        await audit_ctx.record_recommendation(verdict)
```

### 5.5 Short-circuit, but do not cancel

A confirmed watchlist hit is `HOLD` at risk 100, published immediately. **Do not cancel
the running branches.** The officer is about to write an incident report and the full
forensic package is the evidence that report needs. Publish the decision early, keep
collecting in the background, seal everything.

---

## 6. Modules

### 6.1 Gates

**Quality** — Laplacian variance > 120, glare clusters < 4% of pixels, effective DPI
≥ 250 after corner detection, all four corners found, rectification residual within
tolerance. Return the *worst* factor and an actionable hint:
`Glare on the machine-readable zone — tilt the document about 15° and rescan.`

**Classify** — document type, ISO-3166 country, version. Start with a small CNN over
MIDV plus template matching on layout geometry; a heuristic on aspect ratio + MRZ format
+ detected field positions gets you surprisingly far and is explainable. Confidence
< 0.7 → `UNKNOWN_DOCUMENT_TYPE`, route to manual, still run type-agnostic checks.

### 6.2 OCR branch

1. `mrz.detect.find_mrz` → band, deskew, split lines
2. `MRZReader` → checksum-constrained decode → fields + `MrzLine` groups
3. RapidOCR on the VIZ region → text boxes with confidence
4. Field classifier: `[normalised_bbox, text_embedding, template_prior]` → field label.
   A 3-layer MLP or LightGBM trained on MIDV annotations. Trains in minutes.
5. **MRZ ↔ VIZ cross-check** — compare `birth_date`, `expiry_date`, `doc_number`,
   `surname`, `nationality`, `sex`. Any mismatch → `MRZ_VIZ_MISMATCH`, high severity.

Step 5 is the cheapest high-value fraud signal in the entire system. No model, no
training data. Implement it on day one.

### 6.3 Forensics branch

Internally parallel — each technique is a pure function `(image, ctx) -> list[Signal]`
so they fan out across the process pool:

`metadata` (EXIF, quantization tables, double-JPEG) · `ela` · `noise_residual` (SRM) ·
`copy_move` (ORB self-matching) · `font_metrics` (stroke width, x-height, baseline,
kerning per field) · `portrait_seam` · `guilloche_fft` · `print_origin` (halftone screen
ruling and angle) · `tamper_unet` (learned heatmap)

Every one returns normalised regions. A technique that cannot localise returns a signal
with `region=None` and the frontend renders it honestly as non-spatial.

### 6.4 Face branch

Order matters: **PAD runs first**. On a detected spoof, short-circuit the branch and
return `status='SPOOF'` with `similarity=None`. Never compute similarity against a known
spoof — a passing score on a spoofed input is worse than no score.

Then SCRFD detect → 5-point align → ArcFace embed → cosine similarity. Threshold ~0.38,
**calibrated on a held-out set, loaded from config, never hard-coded**. Document
portraits are printed, scanned, often a decade old — that is a real domain shift from
web-photo training pairs, so calibrate for the document-vs-live setting specifically.

### 6.5 Rules engine (Wave 2)

Pure Python, no ML, exhaustively unit-tested. Check digits (delegate to `mrz.spec`),
ISO-3166 validity, date logic, country-specific document number regex, visa validity
window, stay duration vs visa type maximum, entries remaining, expiry.

Country rules live in `data/country_rules.yaml`, not in code. They change by decree.

### 6.6 Database branch (Wave 2)

Internally parallel, I/O-bound:

- document number → stolen/lost registry
- name + DOB → blacklist, fuzzy (Jaro-Winkler) + phonetic
- **face embedding → 1:N gallery search** via pgvector
- identity graph: prior encounters, name/number conflicts, impossible travel, document
  velocity

**The gallery only works if you write to it.** See §8.3 — this is where teams lose the
multiple-identity capability.

### 6.7 Cross-document join (Stage 5.5)

The fraud class no per-document module can see.

```python
if passport and visa:
    if visa.fields.passport_no != passport.fields.doc_no:
        yield Signal(code='VISA_PASSPORT_MISMATCH', severity='critical', weight=45,
                     detail=f'Visa references passport {visa.fields.passport_no}; '
                            f'the presented passport is {passport.fields.doc_no}.')
```

Also: name consistency, DOB consistency, visa validity contained within passport
validity, visa issue date after passport issue date, print-origin signature comparison.

**Session risk is `max`, never `mean`.** One document at 90 and one at 5 is a risk-90
session, not 47.

### 6.8 Correlation (Stage 6)

Spatially cluster signals with regions at IoU ≥ 0.3. A cluster containing ≥ 3 distinct
modules gets `convergence_group` set and a ×1.4 boost — three independent physical
measurements agreeing on the same coordinates is not coincidence.

The inverse matters as much: a lone low-severity signal with no spatial corroboration is
**discounted** ×0.6. A single ELA blob with clean OCR and clean fonts is a compression
artefact. This is the system's primary false-positive suppression mechanism.

### 6.9 Fusion (Stage 7)

Three layers, in order:

1. **Deterministic overrides** — watchlist hit → HOLD 100; spoof → HOLD; expired →
   SECONDARY minimum; checksum unrecoverable → SECONDARY minimum; cross-doc number
   mismatch → HOLD minimum. Rules beat the model in both directions.
2. **Learned fusion** — module scores + convergence boosts → gradient boosting →
   calibrated 0–100. Calibration (isotonic or Platt) matters more than discrimination:
   the number is shown to a human as if it were a probability.
3. **Confidence and abstention** —
   `f(capture_quality, ocr_field_confidence, module_coverage, distance_from_boundary)`.
   Below 0.65, or inside the ambiguous band → `band='ABSTAIN'`, **`risk=None`**.

Emit `risk=None` for both `ABSTAIN` and `RECAPTURE`. Not `0`, not `-1`. `None`.

---

## 7. API

```
POST   /api/v1/screening              multipart → 201 {session_id, status:'processing'}
WS     /ws/screening/{session_id}     progressive ScreeningEvent stream
POST   /api/v1/screening/{id}/decision {decision, note} → sealed session
GET    /api/v1/screening/{id}          full session (replay)
GET    /api/v1/history?lane=&since=    sealed sessions
GET    /api/v1/health                  model manifest, watchlist age, db state
```

**Decision endpoint rules:**
- Rejects a decision on a sealed session (409).
- **Requires a non-empty note** when `decision != system_band`, or when decision is
  `HOLD` or `REFER`. 422 with a clear message otherwise.
- Records `override=True` when they differ. This is the most valuable data the system
  produces — see §8.4.

**WS:** replay buffer per session so a client reconnecting mid-screening receives every
event it missed, in order. Officers' tablets drop connections.

---

## 8. Persistence

### 8.1 Schema

`sessions` · `documents` · `signals` · `decisions` · `audit_records` ·
`gallery(embedding vector(512), identity_key, session_id, checkpoint, ts)` ·
`watchlist` · `graph_edges`

Index the gallery with HNSW on cosine distance.

### 8.2 Audit chain

Append-only, hash-linked, sealed on officer decision:

```python
record = {
    'session_id': sid,
    'prev_hash': last_hash,                      # chain link
    'timestamp_utc': ...,
    'checkpoint_id': ..., 'officer_id': ...,
    'input_hashes': {'passport.jpg': 'sha256:...'},
    'model_versions': manifest.versions,          # <-- from §1.5
    'module_outputs': {...},                      # every raw score
    'system_recommendation': {'risk': 78, 'band': 'SECONDARY'},
    'officer_decision': 'HOLD',
    'override': True,
    'officer_note': '...',
    'duration_ms': 1620,
}
record['hash'] = sha256(canonical_json(record))
```

Model versions are not optional. Six months later, "the system said 78" is meaningless
unless the exact model that produced it can be identified and re-run.

### 8.3 Gallery enrolment — do not skip this

```python
if decision in ('CLEAR', 'SECONDARY'):
    await gallery.upsert(embedding=face.live_embedding,
                         identity_key=(doc_type, country, doc_no),
                         session_id=sid, checkpoint=..., ts=now())
```

On the first encounter the 1:N search returns nothing — the gallery is empty. The
conflict only appears on the *second* encounter under a different name. **The system
only becomes capable of detecting multiple identities after it has been running.**
Seed the gallery in `scripts/seed_gallery.py` so the capability is demonstrable.

### 8.4 Feedback

Officer clears a high-risk session → probable false positive → threshold review.
Officer holds a low-risk session → probable false negative → the highest-value training
example you will ever get. Store both in a review queue.

### 8.5 Retention

Live face frame of a cleared traveller: purge on a short cycle — it is biometric data
belonging to an innocent person. Embeddings: retained per policy, pseudonymous.
Flagged cases: investigation hold. Audit records: statutory retention, immutable.
The purge job itself is logged in the chain.

---

## 9. Build order

**M1 — skeleton and contracts.** FastAPI app, Pydantic contracts ported from
`FRONTEND_BRIEF.md` §3, WS event bus with replay buffer, model registry with hash
verification, docker-compose (Postgres + pgvector + MinIO). Every branch a stub that
sleeps its budget and returns fixture data. *The frontend can integrate at the end of
M1.* That is the point.

**M2 — gates and OCR.** Quality gate, classifier, vendored `mrz/` wired in, RapidOCR for
VIZ, field classifier, MRZ↔VIZ cross-check, `MrzLine` groups for the ribbon.

**M3 — the DAG for real.** Wave 1 with timeouts and `Degraded`, process pool for
forensics, wave 2, cross-document join, correlation, fusion, abstention.

**M4 — forensics and face.** Metadata, ELA, noise, copy-move, font metrics, portrait
seam. SCRFD + PAD + ArcFace. Learned heatmap last — the classical techniques carry most
of the value.

**M5 — persistence.** Audit chain, gallery enrolment, identity graph, watchlist, history
endpoints, retention job.

**M6 — the fifteen cases end to end.** Every fixture in `FRONTEND_BRIEF.md` §6 must be
reproducible from real inputs, not replayed tapes.

### Acceptance

- M1: frontend renders a full session against the real backend.
- M2: ICAO reference MRZ parses; a tampered DOB returns `CHECKSUM_UNRECOVERABLE`.
- M3: killing the face branch mid-request yields a degraded session with
  `no_biometric`, never a clean pass.
- M4: case 3 produces a convergence group of ≥ 3 modules on one region.
- M5: audit chain verifies end to end; second encounter under a different name fires
  `IDENTITY_CONFLICT`.
- M6: p95 latency < 2.5 s on the target hardware.

---

## 10. Do not

- Do not let a VLM write an identity field value. §1.4.
- Do not use Surya's weights. §1.2.
- Do not run forensics on the rectified image. §5.2.
- Do not download models at runtime.
- Do not return `0` or `-1` for a withheld risk score. Return `None`.
- Do not treat a database timeout as "no watchlist hit".
- Do not cancel in-flight branches on a short-circuit HOLD.
- Do not hard-code the face threshold.
- Do not put country validation rules in Python.
- Do not use real or scraped identity documents anywhere, including tests.
- Do not add Celery, Kafka, or a microservice split in v1.

---

## 11. Ask before you build

- Whether a green-lane auto-clear policy is ever enabled (changes §0 rule 1).
- Which watchlist source is authoritative and what its sync protocol is.
- Whether NFC/ICAO chip reading is in scope for v1 (it is where the real ground truth
  lives for modern passports, and it changes the trust model substantially).

Everything else is decided. Build it.
