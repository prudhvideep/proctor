"""
Proctor Chat — Gradio Demo

A privacy layer between the user and LLMs:
  1. Voice/text/document input
  2. Agentic PII scan (FunctionGemma on-device)
  3. HITL approval with before/after comparison
  4. Redact PII → route to local or cloud LLM
  5. Get text response → rehydrate PII → display

Usage:
    DEMO_MODE=1 GEMINI_API_KEY=your-key python demo.py
"""

import sys
sys.path.insert(0, "python/src")

import os, time

os.environ["DEMO_MODE"] = "1"

import gradio as gr
from pii_utils import redact, rehydrate, PIIVault

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import docx  # python-docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

from main import (
    detect_pii,
    generate_local_text, generate_cloud_text,
    should_route_text_to_cloud,
    select_extraction_tool,
    select_agent_tools, execute_tool,
    transcribe_audio,
)


# --- File extraction ---

def extract_text_from_file(file_path: str) -> str:
    if file_path is None:
        return ""
    lower = file_path.lower()
    if lower.endswith(".pdf"):
        if not HAS_PYMUPDF:
            return "[Error: PyMuPDF not installed — pip install pymupdf]"
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    if lower.endswith(".docx"):
        if not HAS_DOCX:
            return "[Error: python-docx not installed — pip install python-docx]"
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    # Plain text fallback (.txt, .md, .csv, etc.)
    with open(file_path, "r", errors="replace") as f:
        return f.read()


# --- Privacy scorecard helpers ---

def _init_stats(state):
    """Ensure stats keys exist in state."""
    state.setdefault("stats_queries", 0)
    state.setdefault("stats_pii_caught", 0)
    state.setdefault("stats_on_device", 0)
    state.setdefault("stats_cloud", 0)
    state.setdefault("stats_pii_to_cloud", 0)


def _render_scorecard(state):
    """Render the privacy scorecard markdown."""
    total = state.get("stats_queries", 0)
    pii = state.get("stats_pii_caught", 0)
    on_device = state.get("stats_on_device", 0)
    cloud = state.get("stats_cloud", 0)
    leaked = state.get("stats_pii_to_cloud", 0)
    ratio = f"{on_device / total * 100:.0f}%" if total > 0 else "—"

    return (
        f"### Privacy Scorecard\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Queries processed | **{total}** |\n"
        f"| PII entities caught | **{pii}** |\n"
        f"| Routed on-device | **{on_device}** |\n"
        f"| Routed to cloud | **{cloud}** |\n"
        f"| On-device ratio | **{ratio}** |\n"
        f"| PII leaked to cloud | **{leaked}** |\n"
    )


# --- Pipeline execution ---

def _execute_pipeline(state, redacted: bool, no_pii: bool = False):
    """Route query to local or cloud LLM, rehydrate PII in response."""
    original = state["original"]
    scan_ms = state.get("scan_ms", 0)
    extraction_info = state.get("extraction_info")
    transcribe_ms = state.get("transcribe_ms", 0)

    vault = PIIVault()
    query_to_send = original
    if redacted and not no_pii:
        query_to_send = state.get("redacted", original)
        vault.mappings = state.get("vault_mappings", {})

    # --- Agentic tool selection (on-device) ---
    tool_info = select_agent_tools(query_to_send)
    tool_calls = tool_info.get("tool_calls", [])
    tool_select_ms = tool_info.get("select_ms", 0)

    # Execute selected tools
    tool_results = {}
    for call in tool_calls:
        name = call.get("name")
        args = call.get("arguments", {})
        tool_results[name] = execute_tool(name, args)

    # Build enhanced query with tool results as context
    llm_query = query_to_send
    if tool_results:
        context_lines = [f"[{name}]: {result}" for name, result in tool_results.items()]
        llm_query = f"{query_to_send}\n\nContext from tools:\n" + "\n".join(context_lines)

    # Routing decision based on query complexity
    use_cloud = should_route_text_to_cloud(original)

    # Update stats
    _init_stats(state)
    state["stats_queries"] += 1
    if use_cloud:
        state["stats_cloud"] += 1
    else:
        state["stats_on_device"] += 1

    exec_start = time.time()

    if use_cloud:
        result = generate_cloud_text(llm_query)
        source = "Cloud"
        route_reason = "Complex query — routed to cloud"
    else:
        result = generate_local_text(llm_query)
        source = "On-device"
        route_reason = "Simple query — handled locally"

    exec_ms = (time.time() - exec_start) * 1000
    response_text = result.get("response", "")
    model_name = result.get("model", "Unknown")

    # Rehydrate PII in the response
    rehydrated = False
    if redacted and vault.mappings and response_text:
        original_response = response_text
        response_text = rehydrate(response_text, vault)
        rehydrated = (response_text != original_response)

    # Build the output
    total_ms = transcribe_ms + scan_ms + tool_select_ms + exec_ms
    entity_count = len(state.get("entities", []))

    # --- LLM Response ---
    lines = []
    if response_text:
        lines.append(response_text)
    else:
        lines.append("*No response generated.*")

    # --- Pipeline Trace ---
    extract_ms = extraction_info.get("extract_ms", 0) if extraction_info else 0
    total_ms += extract_ms
    lines.append(f"\n---\n**Pipeline Trace** ({total_ms:.0f}ms total)")

    step = 1

    # Voice transcription step
    if transcribe_ms > 0:
        lines.append(f"\n**{step}. Voice Transcription** ({transcribe_ms:.0f}ms, on-device)")
        lines.append(f"```\nTool Call → cactus_transcribe(model=\"whisper-small\")\n```")
        step += 1

    # Document extraction step
    if extraction_info:
        tool_name = extraction_info.get("tool_called") or "extract_text_file"
        lines.append(f"\n**{step}. Document Extraction** ({extract_ms:.0f}ms, on-device)")
        lines.append(f"```\nTool Call → {tool_name}()\n```")
        step += 1

    pii_summary = "No PII found" if no_pii else f"{entity_count} entities flagged"
    lines.append(f"\n**{step}. PII Detection** ({scan_ms:.0f}ms, on-device)")
    lines.append(f"```\nTool Call → flag_pii() — {pii_summary}\n```")
    step += 1

    if not no_pii:
        action = "Approved — PII redacted" if redacted else "Skipped — sent as-is"
        lines.append(f"\n**{step}. HITL Review** — {action}")
        step += 1
        if redacted and vault.mappings:
            lines.append(f"\n**{step}. Redaction** — {len(vault.mappings)} placeholder(s) inserted")
            step += 1

    # Tool selection + execution step
    lines.append(f"\n**{step}. Tool Selection** ({tool_select_ms:.0f}ms, on-device)")
    if tool_calls:
        tool_lines = []
        for call in tool_calls:
            name = call.get("name")
            args = call.get("arguments", {})
            arg_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
            tool_lines.append(f"Tool Call → {name}({arg_str})")
            if name in tool_results:
                result_preview = str(tool_results[name])[:100]
                tool_lines.append(f"Result   → {result_preview}")
        lines.append("```\n" + "\n".join(tool_lines) + "\n```")
    else:
        lines.append("```\nNo tools needed for this query\n```")
    step += 1

    route_tool = "route_to_cloud" if use_cloud else "route_to_local"
    lines.append(f"\n**{step}. Routing Decision**")
    lines.append(f"```\nTool Call → {route_tool}() — {route_reason}\n```")
    step += 1

    lines.append(f"\n**{step}. LLM Response** ({exec_ms:.0f}ms, {source})")
    lines.append(f"```\nModel → {model_name} ({source})\n```")
    step += 1

    if rehydrated:
        lines.append(f"\n**{step}. Rehydration** — PII restored in response")
        lines.append(f"```\nTool Call → rehydrate_pii() — {len(vault.mappings)} placeholder(s) swapped back\n```")
        step += 1

    if redacted and vault.mappings and "Cloud" in source:
        lines.append(f"\n> **Privacy:** Cloud only saw redacted query. PII vault stayed on-device.")

    return "\n".join(lines)


# --- Chat handler ---

def respond(message, chat_history, state, file_upload, audio):
    """Main chat handler — privacy-first LLM pipeline."""
    _init_stats(state)

    # --- Handle voice input ---
    transcribe_ms = 0
    if audio is not None:
        tr = transcribe_audio(audio)
        transcribed_text = tr.get("text", "")
        transcribe_ms = tr.get("transcribe_ms", 0)

        if transcribed_text:
            chat_history.append({"role": "assistant", "content": (
                f"**Step 0 · Voice Transcription** — Whisper on-device ({transcribe_ms:.0f}ms)\n\n"
                f"```\n"
                f"Tool Call → cactus_transcribe(model=\"whisper-small\")\n"
                f"Result   → \"{transcribed_text}\"\n"
                f"```"
            )})
            if message.strip():
                message = message.strip() + " " + transcribed_text
            else:
                message = transcribed_text

    if not message.strip() and file_upload is None:
        return "", chat_history, state, _render_scorecard(state), None

    # --- Handle HITL approval ---
    if state.get("phase") == "awaiting_approval":
        lower = message.strip().lower()

        if lower in ("approve", "yes", "y"):
            chat_history.append({"role": "user", "content": message})
            state["stats_pii_caught"] += len(state.get("entities", []))
            result_msg = _execute_pipeline(state, redacted=True)
            chat_history.append({"role": "assistant", "content": result_msg})
            state["phase"] = "idle"
            return "", chat_history, state, _render_scorecard(state), None

        elif lower in ("skip", "no", "n"):
            chat_history.append({"role": "user", "content": message})
            result_msg = _execute_pipeline(state, redacted=False)
            chat_history.append({"role": "assistant", "content": result_msg})
            state["phase"] = "idle"
            return "", chat_history, state, _render_scorecard(state), None

        else:
            chat_history.append({"role": "user", "content": message})
            chat_history.append({"role": "assistant", "content": "Type **approve** to redact PII before sending to the LLM, or **skip** to send as-is."})
            return "", chat_history, state, _render_scorecard(state), None

    # --- New query ---
    query = message.strip()
    extraction_info = None
    if file_upload is not None:
        filename = os.path.basename(file_upload.name)

        # Agentic tool selection: FunctionGemma picks the extraction tool
        extraction_info = select_extraction_tool(filename)
        tool_called = extraction_info.get("tool_called")
        extract_ms = extraction_info.get("extract_ms", 0)

        # Execute the extraction
        extracted = extract_text_from_file(file_upload.name)

        # Show the agentic extraction step in chat
        tool_label = tool_called or "extract_text_file"
        chat_history.append({"role": "assistant", "content": (
            f"**Step 1 · Document Extraction Agent** — FunctionGemma on-device ({extract_ms:.0f}ms)\n\n"
            f"```\n"
            f"Tool Call → {tool_label}(file=\"{filename}\")\n"
            f"Result   → Extracted {len(extracted.split())} words\n"
            f"```\n\n"
            f"Scanning extracted text for PII..."
        )})

        if extracted and not query:
            query = extracted
        elif extracted:
            query = query + "\n\n" + extracted

    if not query:
        return "", chat_history, state, _render_scorecard(state), None

    chat_history.append({"role": "user", "content": query})

    # --- Agentic PII detection (on-device) ---
    scan_start = time.time()
    entities = detect_pii(query)
    scan_ms = (time.time() - scan_start) * 1000

    # Build step label (Step 1 if no doc, Step 2 if doc was extracted)
    pii_step = 2 if extraction_info else 1
    if transcribe_ms > 0:
        pii_step += 1

    if entities:
        redacted_text, vault = redact(query, entities)

        state["phase"] = "awaiting_approval"
        state["original"] = query
        state["redacted"] = redacted_text
        state["vault_mappings"] = vault.mappings
        state["entities"] = [(e.entity_type, e.value) for e in entities]
        state["scan_ms"] = scan_ms
        state["extraction_info"] = extraction_info
        state["transcribe_ms"] = transcribe_ms

        # Show tool calls + clear entity → placeholder mapping
        lines = [
            f"**Step {pii_step} · PII Detection Agent** — FunctionGemma on-device ({scan_ms:.0f}ms)\n",
            "```",
        ]
        for e in entities:
            lines.append(f'Tool Call → flag_pii(entity="{e.value}", entity_type="{e.entity_type}")')
        lines.append("```")

        lines.append(f"\n{len(entities)} PII entities flagged.\n")

        # Clear mapping: what was found → what it gets replaced with
        lines.append("| PII Found | Type | Replaced With |")
        lines.append("|-----------|------|---------------|")
        for entity in vault.entities:
            lines.append(f"| `{entity.value}` | {entity.entity_type} | `{entity.placeholder}` |")

        lines.append(f"\n---\n**Step {pii_step + 1} · Human-in-the-Loop Review**")
        lines.append("Type **approve** to redact PII before sending, or **skip** to send as-is.")

        chat_history.append({"role": "assistant", "content": "\n".join(lines)})
        return "", chat_history, state, _render_scorecard(state), None

    else:
        # No PII — route directly
        state["original"] = query
        state["scan_ms"] = scan_ms
        state["entities"] = []
        state["extraction_info"] = extraction_info
        state["transcribe_ms"] = transcribe_ms

        status = (
            f"**Step {pii_step} · PII Detection Agent** — FunctionGemma on-device ({scan_ms:.0f}ms)\n\n"
            f"```\n"
            f"Tool Call → flag_pii() — no PII entities found\n"
            f"```\n\n"
            f"No PII detected. Routing query directly..."
        )
        chat_history.append({"role": "assistant", "content": status})

        result_msg = _execute_pipeline(state, redacted=False, no_pii=True)
        chat_history.append({"role": "assistant", "content": result_msg})
        state["phase"] = "idle"
        return "", chat_history, state, _render_scorecard(state), None


# --- Build the app ---

EXAMPLE_QUERIES = [
    "What medication should I recommend for patient John Doe who has the flu?",
    "Send a message to prudhvi about his common cold medication",
    "What's the weather like today?",
    "Explain how photosynthesis works",
    "Patient Jane Smith, SSN 123-45-6789, needs a referral for her diabetes",
]


def build_app():
    with gr.Blocks(
        title="Proctor",
        css="footer { display: none !important; } .progress-bar, .meta-text, .progress-text { display: none !important; }",
    ) as app:
        gr.Markdown(
            "# Proctor\n"
            "A privacy layer between you and LLMs. "
            "PII is detected on-device, reviewed by you, and redacted before leaving your machine. "
            "Simple queries stay local; complex ones go to the cloud with PII stripped."
        )

        state = gr.State({"phase": "idle"})

        with gr.Row():
            # --- Main chat column ---
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(height=500, show_label=False)

                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask anything... PII will be detected and redacted automatically.",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

                gr.Examples(
                    examples=[[q] for q in EXAMPLE_QUERIES],
                    inputs=[msg],
                    label="Try these",
                )

                with gr.Row():
                    with gr.Accordion("Upload Document", open=False):
                        file_upload = gr.File(
                            label="PDF / DOCX / TXT",
                            file_types=[".pdf", ".docx", ".txt", ".md", ".csv"],
                        )
                    with gr.Accordion("Voice Input", open=False):
                        audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Record and send",
                        )

            # --- Sidebar: Privacy Scorecard ---
            with gr.Column(scale=1, min_width=250):
                scorecard = gr.Markdown(_render_scorecard({"phase": "idle"}))

                gr.Markdown(
                    "---\n"
                    "### How it works\n\n"
                    "1. **Input** — text, voice, or document\n"
                    "2. **PII Scan** — FunctionGemma on-device\n"
                    "3. **HITL Review** — you approve or skip\n"
                    "4. **Redact** — PII replaced with placeholders\n"
                    "5. **Route** — simple → local, complex → cloud\n"
                    "6. **Respond** — LLM generates answer\n"
                    "7. **Rehydrate** — PII restored in response\n"
                )

        submit_args = dict(
            fn=respond,
            inputs=[msg, chatbot, state, file_upload, audio_input],
            outputs=[msg, chatbot, state, scorecard, audio_input],
        )
        msg.submit(**submit_args)
        send_btn.click(**submit_args)

        gr.Markdown(
            "---\n*Google DeepMind × Cactus Compute Hackathon* · "
            "FunctionGemma on-device · Gemini cloud fallback"
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
