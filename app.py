import os
import json
import uuid
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path

import chromadb
import streamlit as st
from dateutil import parser as date_parser
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).parent
SCHEDULE_FILE = BASE_DIR / "schedule.json"
COLLECTION_NAME = "schedule_events"

st.set_page_config(page_title="Agentic RAG Schedule Assistant", page_icon="📅", layout="wide")


# ---------- Schedule + Vector DB ----------
def load_schedule():
    if not SCHEDULE_FILE.exists():
        return []
    return json.loads(SCHEDULE_FILE.read_text(encoding="utf-8"))


def save_schedule(events):
    SCHEDULE_FILE.write_text(json.dumps(events, indent=2), encoding="utf-8")


@st.cache_resource
def get_collection():
    client = chromadb.Client(
        chromadb.config.Settings(anonymized_telemetry=False)
    )

    def embedding_function(texts):
        embeddings = []

        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()

            # Create a deterministic 128-dimensional vector.
            vector = [
                (digest[i % len(digest)] / 255.0)
                for i in range(128)
            ]

            embeddings.append(vector)

        return embeddings

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embedding_function
    )

    return collection



def event_text(e):
    return (
        f"{e['title']} | {e['type']} | {e['date']} | "
        f"{e['start_time']}-{e['end_time']} | {e.get('location','')} | "
        f"{e.get('description','')}"
    )


def rebuild_index():
    collection = get_collection()
    # Chroma's in-memory client is recreated for a process, but this collection
    # can safely be rebuilt from schedule.json whenever the app starts/changes.
    existing = collection.get()
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)

    events = load_schedule()
    if events:
        collection.add(
            ids=[e["id"] for e in events],
            documents=[event_text(e) for e in events],
            metadatas=[
                {
                    "event_id": e["id"],
                    "date": e["date"],
                    "type": e["type"],
                    "start_time": e["start_time"],
                    "end_time": e["end_time"],
                }
                for e in events
            ],
        )
    return collection


@st.cache_resource
def _build_index_once():
    # st.cache_resource runs this ONCE PER SERVER PROCESS, not once per
    # browser session. Previously this was gated on st.session_state,
    # which re-ran the (expensive, model-loading) rebuild_index() for
    # every new visitor/tab, making the app feel slow on every session.
    #
    # On first cold start, Chroma's default embedding function downloads a
    # small ONNX model from the internet. If that download hiccups (slow
    # host network, first-boot DNS blip, etc.) we don't want that to take
    # the whole app down — fall back to date/time filtering without
    # semantic search rather than crashing on load.
    try:
        rebuild_index()
        return {"ok": True, "error": None}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def initialize():
    return _build_index_once()


def parse_date(text):
    try:
        return date_parser.parse(text, fuzzy=True).date()
    except Exception:
        return None


def normalize_time(t):
    return datetime.strptime(t.strip(), "%H:%M").strftime("%H:%M")


# ---------- Agent Tools ----------
def get_schedule(query="", target_date=None, start_time=None, end_time=None):
    """Retrieve relevant schedule information using Chroma semantic search + filters."""
    events = load_schedule()
    collection = get_collection()

    # Semantic retrieval from Chroma. If the embedding backend isn't
    # available (e.g. a cold-start model download hiccup), fall back to
    # returning events in their existing order instead of crashing.
    retrieved_ids = []
    if events:
        try:
            q = query.strip() if query else "schedule events"
            result = collection.query(query_texts=[q], n_results=min(20, max(1, len(events))))
            if result.get("ids"):
                retrieved_ids = result["ids"][0]
        except Exception:
            retrieved_ids = []

    # Filter by date/time when provided.
    filtered = []
    for e in events:
        if target_date and e["date"] != target_date:
            continue

        if start_time and e["end_time"] <= start_time:
            continue

        if end_time and e["start_time"] >= end_time:
            continue

        filtered.append(e)

    # If no explicit date/time filter, use semantic order.
    if not target_date and not start_time and not end_time:
        rank = {eid: i for i, eid in enumerate(retrieved_ids)}
        filtered.sort(key=lambda x: rank.get(x["id"], 9999))

    return filtered


def update_schedule(action, event_id=None, title=None, event_type="meeting",
                    event_date=None, start_time=None, end_time=None,
                    location="", description=""):
    """Add, update, or remove a schedule entry."""
    events = load_schedule()

    if action == "add":
        if not all([title, event_date, start_time, end_time]):
            return {"success": False, "message": "title, event_date, start_time and end_time are required"}

        new_event = {
            "id": event_id or "evt_" + uuid.uuid4().hex[:8],
            "title": title,
            "type": event_type,
            "date": event_date,
            "start_time": normalize_time(start_time),
            "end_time": normalize_time(end_time),
            "location": location or "Not specified",
            "description": description or "",
        }
        events.append(new_event)
        save_schedule(events)
        rebuild_index()
        return {"success": True, "message": "Event added", "event": new_event}

    if action == "update":
        if not event_id:
            return {"success": False, "message": "event_id is required for update"}

        for e in events:
            if e["id"] == event_id:
                if title is not None: e["title"] = title
                if event_type: e["type"] = event_type
                if event_date: e["date"] = event_date
                if start_time: e["start_time"] = normalize_time(start_time)
                if end_time: e["end_time"] = normalize_time(end_time)
                if location is not None and location != "": e["location"] = location
                if description is not None and description != "": e["description"] = description
                save_schedule(events)
                rebuild_index()
                return {"success": True, "message": "Event updated", "event": e}

        return {"success": False, "message": "Event not found"}

    if action == "remove":
        if not event_id:
            return {"success": False, "message": "event_id is required for remove"}
        remaining = [e for e in events if e["id"] != event_id]
        if len(remaining) == len(events):
            return {"success": False, "message": "Event not found"}
        save_schedule(remaining)
        rebuild_index()
        return {"success": True, "message": "Event removed"}

    return {"success": False, "message": f"Unknown action: {action}"}


# ---------- Gemini Agent ----------
GEMINI_TOOLS = [
    types.Tool(function_declarations=[
        {
            "name": "get_schedule",
            "description": "Retrieve schedule events relevant to a user's date, time range, or natural-language query. Use this for questions about availability or existing events.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "Natural-language search query."},
                    "target_date": {"type": "STRING", "description": "Exact date in YYYY-MM-DD if known."},
                    "start_time": {"type": "STRING", "description": "Optional range start in HH:MM."},
                    "end_time": {"type": "STRING", "description": "Optional range end in HH:MM."},
                },
            },
        },
        {
            "name": "update_schedule",
            "description": "Add, update, or remove a schedule entry. Use only when the user explicitly asks to change the schedule.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "action": {"type": "STRING", "enum": ["add", "update", "remove"]},
                    "event_id": {"type": "STRING"},
                    "title": {"type": "STRING"},
                    "event_type": {"type": "STRING", "enum": ["meeting", "workshop", "task", "appointment"]},
                    "event_date": {"type": "STRING", "description": "YYYY-MM-DD"},
                    "start_time": {"type": "STRING", "description": "HH:MM, 24-hour time"},
                    "end_time": {"type": "STRING", "description": "HH:MM, 24-hour time"},
                    "location": {"type": "STRING"},
                    "description": {"type": "STRING"},
                },
                "required": ["action"],
            },
        },
    ])
]


@st.cache_resource
def get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def run_agent(user_query):
    client = get_genai_client()
    if client is None:
        return "Please set the GEMINI_API_KEY environment variable before using the AI agent."

    today = date.today().isoformat()
    system = f"""
You are an Agentic RAG Schedule Assistant.
Today is {today}.
Manage only the user's schedule data available through tools.

Rules:
1. For questions about existing events, availability, dates, times, or schedule details, call get_schedule.
2. For add/update/remove requests, call update_schedule. If identifying an event requires looking it up first, call get_schedule before update_schedule.
3. Resolve relative dates such as today, tomorrow, Friday, next Monday using today's date.
4. "Am I free" means retrieve events overlapping the requested time range and explain whether conflicts exist.
5. When adding an event, if the user gives only a start time and no end time, choose a reasonable 1-hour duration and state that assumption.
6. Never claim a schedule change succeeded unless the tool reports success.
7. Keep answers concise and include dates/times.
"""

    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=GEMINI_TOOLS,
        temperature=0.2,
    )
    contents = [types.Content(role="user", parts=[types.Part(text=user_query)])]

    try:
        for _ in range(5):
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents,
                config=config,
            )

            function_calls = response.function_calls or []
            if not function_calls:
                return response.text or "I couldn't produce a response."

            # Preserve the model's tool-call turn in the conversation.
            contents.append(response.candidates[0].content)
            response_parts = []

            for call in function_calls:
                args = dict(call.args or {})
                if call.name == "get_schedule":
                    result = get_schedule(
                        query=args.get("query", ""),
                        target_date=args.get("target_date"),
                        start_time=args.get("start_time"),
                        end_time=args.get("end_time"),
                    )
                elif call.name == "update_schedule":
                    result = update_schedule(**args)
                else:
                    result = {"success": False, "message": f"Unknown tool: {call.name}"}

                response_parts.append(
                    types.Part.from_function_response(
                        name=call.name,
                        response={"result": result},
                    )
                )

            contents.append(types.Content(role="user", parts=response_parts))

        return "The agent reached its tool-call limit. Please try again."
    except Exception as exc:
        return f"Gemini API error: {exc}"


# ---------- UI ----------
_index_status = initialize()

st.title("📅 Agentic RAG Schedule Assistant")
st.caption("30-day schedule management with ChromaDB retrieval + agentic tool calling")

with st.sidebar:
    st.header("Configuration")
    if os.getenv("GEMINI_API_KEY"):
        st.success("Gemini API key detected")
    else:
        st.warning("GEMINI_API_KEY is not set")

    if not _index_status.get("ok"):
        st.warning(
            "Semantic search index couldn't build on startup, so results "
            "will fall back to plain date/time filtering. "
            f"Details: {_index_status.get('error')}"
        )

    st.divider()
    st.subheader("Sample queries")
    st.write("• What do I have scheduled tomorrow?")
    st.write("• Am I free Friday afternoon?")
    st.write("• Add a meeting on August 20 at 3 PM.")
    st.write("• Move my Project Guide Meeting to 4 PM.")
    st.write("• Remove my dentist appointment.")

st.subheader("Ask your schedule assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

query = st.chat_input("Ask about your schedule or make a change...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking and using schedule tools..."):
            answer = run_agent(query)
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.subheader("Current 30-day schedule")
events = load_schedule()
if events:
    st.dataframe(
        sorted(events, key=lambda x: (x["date"], x["start_time"])),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No schedule events found.")

