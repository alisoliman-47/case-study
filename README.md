# case-study
Sanofi Case Study

## Architecture

```mermaid
flowchart LR
    Q[User Query] --> R[Retriever<br/>MiniLM embeddings + cosine]
    R -->|Top-K abstracts| S[Summarizer<br/>t5-small (CPU)]
    S -->|2–3 sentence summary + keywords| OUT[(Report: CSV & JSON)]
    S --> V[(Optional Verifier<br/>Theme assignment)]
    V --> OUT
```

Retriever: sentence-transformers/all-MiniLM-L6-v2

Summarizer: t5-small (distilled BART fallback is supported)

Verifier: labels like Deep Learning / Clinical Trial / Traditional Methods

##  How to Run

> Requirements: Python 3.9+ (3.10 recommended), ~2 GB free disk, internet access to read the public S3 bucket.

### 1) Create the environment
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install "transformers>=4.42" "torch>=2.2" "sentence-transformers>=2.2" \
            s3fs "pandas>=2.2" "numpy>=1.26" "scikit-learn>=1.4" "tqdm>=4.66"
jupyter notebook

## Design Choices & Trade-offs

Scope & runtime

< 100 docs to guarantee fast, CPU-only runs and predictable demo time.

Seeded sampling for reproducibility between runs/interviews.

Retrieval

MiniLM (all-MiniLM-L6-v2) embeddings: strong semantic signal at low cost.

Simple cosine similarity on normalized vectors (NumPy) → no extra infra to operate.

Trade-off: not as scalable as a vector DB; acceptable for a prototype. Easy to swap in FAISS/pgvector later.

Parsing

Heuristic title/abstract extractor tailored to PMC TXT (handles “Front” blocks, content-type lines, true Abstract headers).

Trade-off: occasional edge cases remain; avoids heavy PDF parsing to keep things lightweight.

Summarization

Default t5-small (with distilled BART fallback) for 2–3 sentence summaries.

Trade-off: smaller models are faster but less nuanced than larger LLMs; acceptable here given the scope.

Keywords

Per-doc TF-IDF for quick topical cues; no training required.

Trade-off: purely lexical; not concept-aware, but cheap and effective for scanability.

Verifier (optional)

Simple theme tags (e.g., Deep Learning / Clinical Trial / Traditional Methods) via zero-shot or similarity.

Trade-off: coarse and not production-grade; useful for triage, can be replaced by a fine-tuned classifier.

Simplicity & infra

In-memory index rather than a vector DB; fewer moving parts, clearer notebook.

Direct S3 read (s3fs) to avoid pre-syncing large datasets.

Trade-off: listing can be blocked on some networks → provide AWS-CLI/local fallback.

Outputs & reviewability

Emit both CSV and JSON for easy inspection, grading, or downstream tooling.

Include retrieval scores to explain ranking decisions.
