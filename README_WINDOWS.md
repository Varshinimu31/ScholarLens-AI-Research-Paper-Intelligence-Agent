# ScholarLens AI — Research Paper Intelligence & Literature Review Agent

**Stack:** Python, FastAPI, arXiv API, OpenAlex API, Ollama LLM, RAG, React/Vite frontend.

## Windows 11 — step-by-step

### 1. Prerequisites
- Python **3.11 or 3.12** (recommended: 3.12; avoid 3.15 for ML package compatibility)
- Node.js LTS with npm: https://nodejs.org/
- Ollama: https://ollama.com/ ; download the model using `ollama pull llama3.2:3b`
- Internet access for academic API search; local Ollama for analysis.

### 2. Open extracted folder in VS Code
Open the folder `Research_Paper_Intelligence_Agent` (the folder that contains `backend` and `frontend`).

### 3. Terminal 1: Python backend
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
If `py -3.12` is not available, run `py -0p` to check installed Python versions. Use Python 3.11 or 3.12.

Check `http://127.0.0.1:8000/docs` in a browser.

### 4. Terminal 2: React frontend
Open **a second terminal** in the project root:
```powershell
cd frontend
npm install
npm run dev
```
Open `http://127.0.0.1:5173/`.

### 5. Ollama
Make sure the Ollama desktop app is running. If needed, run in another terminal:
```powershell
ollama pull llama3.2:3b
ollama list
```
The backend defaults to `http://127.0.0.1:11434` and model `llama3.2:3b`.

### 6. Test end to end
1. Search `large language models in healthcare`.
2. Select 2–3 returned papers.
3. Open **AI workspace** and click **Generate analysis** for Summarize, Compare, or Research gaps.
4. Choose **Ask research AI**, enter a question and generate a citation-linked answer.
5. If an API source is unavailable, the app will show a provider warning and use the other source if possible.

## Optional semantic embedding retrieval
The default RAG retrieval uses a **lexical fallback** to work without large ML dependencies. To enable semantic embeddings, in backend virtual environment:
```powershell
python -m pip install -r requirements-embeddings.txt
$env:USE_EMBEDDINGS="1"
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
This downloads `all-MiniLM-L6-v2` on first use. When embeddings are not enabled or cannot load, the `/research/ask` endpoint reports `retrieval_mode: lexical`.

## API endpoints
- `GET /health` — service health
- `GET /papers/search?query=...&source=both&limit=8` — search arXiv and OpenAlex; optional `year`, `author` filters
- `GET /papers/{paper_id}` — cached paper details
- `POST /papers/summarize` — abstract-grounded summaries
- `POST /papers/compare` — paper comparisons
- `POST /research/gaps` — research-gap hypotheses
- `POST /research/ask` — retrieved context Q&A with source citations

## Important limitations
- Results are based on **metadata and abstracts**, not full-text PDFs. LLM summaries and gap suggestions must be verified against original papers.
- Author and year filters are applied to retrieved search results, not a complete historical database search.
- Paper data is stored **in server memory** and resets on backend restart. Search again after restarting.
- arXiv and OpenAlex are external services; outages and rate limits may affect results.
- Research paper citations are clickable source links; this is not a verified citation formatting engine.
- React frontend is local-only for this starter project. Production deployment needs authentication, rate limiting, caching and more robust storage.
