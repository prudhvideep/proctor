# Proctor

A hybrid on-device + cloud function-calling system built on the [Cactus](https://github.com/cactus-compute/cactus) inference engine. Proctor routes tool-calling requests intelligently between a local model and a cloud LLM, optimizing for latency and accuracy based on query complexity.

## How it works

Proctor uses two on-device models to handle inference locally whenever possible:

- **FunctionGemma-270M** — handles function/tool calling and PII detection. Runs fully on-device via Cactus.
- **LFM2.5-1.2B** — handles open-ended text generation for chat-style queries.
- **Gemini 2.5 Flash** — cloud fallback, invoked only when local confidence is below threshold or multi-intent is detected.

### Routing logic

Proctor routes each request through one of three paths:

1. **Fast path** (single tool, single intent) → always on-device.
2. **Medium path** (multiple tools, single intent) → on-device with Tool RAG to narrow the candidate set; falls back to cloud if confidence is low.
3. **Hard path** (multi-intent detected) → attempts on-device, escalates to cloud if the local model under-predicts expected call count or confidence is below threshold.

Cloud fallback confidence thresholds scale with tool-set size: `0.5` for 1 tool, `0.7` for 2–3, `0.8` for 4+.

### PII protection

When `DEMO_MODE=1`, Proctor intercepts cloud-bound messages and runs a two-stage PII pipeline before they leave the device:

1. **Agentic detection** — FunctionGemma is prompted to call `flag_pii` for each PII entity (names, emails, SSNs, medical data, etc.).
2. **Regex supplement** — pattern-based detection catches anything the model missed.

Detected entities are redacted with stable pseudonyms before the message is sent to the cloud, then rehydrated in the returned function-call arguments so the caller receives original values.

### Additional capabilities

- **Voice transcription** — on-device Whisper (small) transcribes audio files before routing.
- **Document routing** — FunctionGemma selects the right extraction tool (`extract_pdf`, `extract_docx`, `extract_text_file`) based on filename.

## Project structure

```
main.py          # Core inference engine: routing, PII pipeline, tool calling
pii_utils.py     # PIIVault, regex detectors, redact/rehydrate helpers
demo.py          # Interactive demo with DEMO_MODE PII protection
benchmark.py     # Evaluation harness for the Cactus Evals leaderboard
submit.py        # Submits main.py to the leaderboard server
setup            # Environment setup script
```

## Setup

```bash
# Install dependencies and download weights
./setup

# Run the demo (enables PII pre-pass)
DEMO_MODE=1 python demo.py

# Run the benchmark
python benchmark.py

# Submit to the leaderboard
python submit.py --team "YourTeamName" --location "SF"
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | For cloud fallback | Gemini API key for cloud inference and web search |
| `DEMO_MODE` | No (default `0`) | Set to `1` to enable PII redaction before cloud calls |
