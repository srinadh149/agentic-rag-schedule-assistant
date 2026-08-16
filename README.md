# Agentic RAG Schedule Assistant

A Streamlit application that manages a user's schedule for the next 30 days using:
- ChromaDB vector database for retrieval
- RAG-style semantic search
- Gemini tool calling for an agentic workflow
- Two tools: `get_schedule` and `update_schedule`
- Add, update, and remove schedule entries

Streamlit already serves as both the UI and the app logic here — there's no
separate frontend/backend split to build or deploy.

## 1. Run locally

Create a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install packages:

```bash
pip install -r requirements.txt
```

Set your API key.

Windows CMD:

```cmd
set GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

macOS/Linux:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Run:

```bash
streamlit run app.py
```

## 2. Test these queries

- What do I have scheduled tomorrow?
- Am I free Friday afternoon?
- Add a meeting on August 20 at 3 PM.
- Move my Project Guide Meeting to 4 PM.
- Remove my dentist appointment.

## 3. Deploy to Render

1. Push this folder to a GitHub repository (include `render.yaml`).
2. In Render, click **New > Blueprint** and point it at the repo — Render
   will read `render.yaml` and set the build/start commands automatically.
   (If you create the service manually instead, set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`)
3. In the service's **Environment** tab, add `GEMINI_API_KEY` with your key.
4. Deploy. Render assigns a `https://<service-name>.onrender.com` URL.

### If the page loads but shows nothing / errors on every message

Check the service logs in Render's dashboard first — the two most common
causes are a missing `GEMINI_API_KEY` env var, or the free-tier instance
still spinning up (Render free services sleep after inactivity and take
~30-50s to wake on the first request).

## Architecture

User -> Streamlit UI -> Gemini Agent -> Tool selection
                                  |-> get_schedule -> ChromaDB -> schedule data
                                  |-> update_schedule -> schedule.json + ChromaDB
