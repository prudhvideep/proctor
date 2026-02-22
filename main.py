
import sys
sys.path.insert(0, "python/src")
functiongemma_path = "weights/functiongemma-270m-it"
lfm_path = "weights/lfm2.5-1.2b-instruct"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy
try:
    from pii_utils import detect_pii_regex, redact, rehydrate, has_pii_signals, PIIVault, PIIEntity
except ImportError:
    # Submission server only has main.py — stubs for benchmark mode
    class PIIVault:
        mappings = {}
    class PIIEntity:
        pass
    detect_pii_regex = lambda *a, **k: []
    redact = lambda text, *a, **k: (text, PIIVault())
    rehydrate = lambda text, *a, **k: text
    has_pii_signals = lambda *a, **k: False

# Toggle for demo mode (enables PII pre-pass). Off by default for benchmark.
DEMO_MODE = os.environ.get("DEMO_MODE", "0") == "1"


# --- Agentic PII Detection via FunctionGemma ---

PII_DETECTION_TOOL = {
    "type": "function",
    "function": {
        "name": "flag_pii",
        "description": (
            "Flag personally identifiable information found in text. "
            "Call once per PII entity. PII includes: person names, emails, "
            "phone numbers, SSNs, medical conditions, addresses, dates of birth, "
            "credit card numbers, IP addresses."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "The exact PII text as it appears in the input",
                },
                "entity_type": {
                    "type": "string",
                    "description": "Category: NAME, EMAIL, PHONE, SSN, MEDICAL, ADDRESS, DOB, CREDIT_CARD, IP_ADDRESS, OTHER",
                },
            },
            "required": ["entity", "entity_type"],
        },
    },
}

PII_SYSTEM_PROMPT = (
    "You are a PII detection agent. Analyze the user's text and call flag_pii "
    "for EACH piece of personally identifiable information you find. "
    "Look for: person names, email addresses, phone numbers, social security numbers, "
    "medical conditions or diagnoses, physical addresses, dates of birth, "
    "credit card numbers, IP addresses, and any other identifying information."
)


def _find_entity_position(text: str, entity: str, used: set) -> int:
    """Find entity position in text, trying exact then case-insensitive."""
    pos = 0
    while True:
        idx = text.find(entity, pos)
        if idx < 0:
            break
        if idx not in used:
            return idx
        pos = idx + 1
    # Case-insensitive fallback
    lower_text, lower_entity = text.lower(), entity.lower()
    pos = 0
    while True:
        idx = lower_text.find(lower_entity, pos)
        if idx < 0:
            break
        if idx not in used:
            return idx
        pos = idx + 1
    return -1


def detect_pii_agentic(text: str) -> list[PIIEntity]:
    """Use FunctionGemma locally to detect PII via the flag_pii tool."""
    model = cactus_init(functiongemma_path)
    raw_str = cactus_complete(
        model,
        [
            {"role": "system", "content": PII_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        tools=[PII_DETECTION_TOOL],
        force_tools=True,
        max_tokens=512,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return []

    entities = []
    used_positions = set()
    for call in raw.get("function_calls", []):
        if call.get("name") != "flag_pii":
            continue
        args = call.get("arguments", {})
        entity_text = args.get("entity", "").strip()
        entity_type = args.get("entity_type", "OTHER").upper()
        if not entity_text:
            continue
        start = _find_entity_position(text, entity_text, used_positions)
        if start < 0:
            continue
        end = start + len(entity_text)
        used_positions.add(start)
        entities.append(PIIEntity(
            entity_type=entity_type,
            value=text[start:end],
            start=start,
            end=end,
        ))

    entities.sort(key=lambda e: e.start)
    return entities


def detect_pii(text: str) -> list[PIIEntity]:
    """Combined PII detection: agentic (primary) + regex (supplement)."""
    agentic_entities = detect_pii_agentic(text)
    regex_entities = detect_pii_regex(text)
    # Merge: keep all agentic, add non-overlapping regex
    merged = list(agentic_entities)
    agentic_spans = [(e.start, e.end) for e in agentic_entities]
    for re_ent in regex_entities:
        overlaps = any(
            not (re_ent.end <= s or re_ent.start >= e)
            for s, e in agentic_spans
        )
        if not overlaps:
            merged.append(re_ent)
    merged.sort(key=lambda e: e.start)
    return merged


# --- Agentic Document Extraction via FunctionGemma ---

EXTRACTION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_pdf",
            "description": "Extract text content from a PDF document file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PDF file",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_docx",
            "description": "Extract text content from a Microsoft Word (.docx) document",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the DOCX file",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_text_file",
            "description": "Read content from a plain text file (.txt, .md, .csv)",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the text file",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
]

EXTRACTION_SYSTEM_PROMPT = (
    "You are a document processing agent. The user wants to extract text from a file. "
    "Call the appropriate extraction tool based on the file type."
)


def select_extraction_tool(filename: str) -> dict:
    """Use FunctionGemma to select the right extraction tool for a file.

    Returns dict with 'tool_called' (str or None) and 'extract_ms' (float).
    """
    model = cactus_init(functiongemma_path)
    start = time.time()
    raw_str = cactus_complete(
        model,
        [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract text from the uploaded file: {filename}"},
        ],
        tools=EXTRACTION_TOOLS,
        force_tools=True,
        max_tokens=128,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)
    elapsed = (time.time() - start) * 1000

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"tool_called": None, "extract_ms": elapsed}

    calls = raw.get("function_calls", [])
    if calls:
        return {
            "tool_called": calls[0].get("name"),
            "arguments": calls[0].get("arguments", {}),
            "extract_ms": elapsed,
        }
    return {"tool_called": None, "extract_ms": elapsed}


# --- Voice Transcription via Whisper ---

WHISPER_MODEL_PATH = "weights/whisper-small"
WHISPER_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"


def transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio on-device using Whisper via Cactus.

    Returns dict with 'text' (str) and 'transcribe_ms' (float).
    """
    from cactus import cactus_transcribe
    model = cactus_init(WHISPER_MODEL_PATH)
    start = time.time()
    raw_str = cactus_transcribe(model, audio_path, prompt=WHISPER_PROMPT)
    cactus_destroy(model)
    elapsed = (time.time() - start) * 1000

    try:
        raw = json.loads(raw_str)
        return {
            "text": raw.get("response", "").strip(),
            "transcribe_ms": elapsed,
        }
    except json.JSONDecodeError:
        return {"text": raw_str.strip(), "transcribe_ms": elapsed}


# --- Utility Tools for Agentic Pipeline ---

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time. Use when the user asks about today's date, day, or current time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information. Use for weather, news, facts, or any real-time data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression to get a numerical result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression like '2 + 2' or 'sqrt(16)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_message",
            "description": "Send a message to a person. Use when the user wants to send, text, or message someone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Name of the person to message",
                    },
                    "content": {
                        "type": "string",
                        "description": "The message content to send",
                    }
                },
                "required": ["recipient", "content"],
            },
        },
    },
]

AGENT_TOOLS_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Analyze the user's request and call the appropriate tool if needed. "
    "Only call tools that are directly relevant to the request."
)


def select_agent_tools(query: str) -> dict:
    """Use FunctionGemma to decide which utility tools to call for a query.

    Returns dict with 'tool_calls' (list) and 'select_ms' (float).
    """
    model = cactus_init(functiongemma_path)
    start = time.time()
    raw_str = cactus_complete(
        model,
        [
            {"role": "system", "content": AGENT_TOOLS_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        tools=AGENT_TOOLS,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)
    elapsed = (time.time() - start) * 1000

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"tool_calls": [], "select_ms": elapsed}

    return {
        "tool_calls": raw.get("function_calls", []),
        "select_ms": elapsed,
    }


def execute_tool(name: str, args: dict) -> str:
    """Execute a utility tool and return the result as a string."""
    if name == "get_current_datetime":
        from datetime import datetime
        return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    if name == "web_search":
        from google import genai
        search_query = args.get("query", "")
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[f"Provide a brief, factual answer: {search_query}"],
        )
        return response.text

    if name == "calculator":
        import math
        expr = args.get("expression", "")
        try:
            allowed = {"__builtins__": {}, "math": math, "sqrt": math.sqrt, "pi": math.pi, "e": math.e}
            return str(eval(expr, allowed))
        except Exception as e:
            return f"Error: {e}"

    if name == "send_message":
        recipient = args.get("recipient", "unknown")
        content = args.get("content", "")
        return f"Message delivered to {recipient}: \"{content}\""

    return f"Unknown tool: {name}"


# --- On-device and Cloud inference ---

def generate_cactus(messages, tools, tool_rag_top_k=None):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    kwargs = dict(
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    if tool_rag_top_k is not None:
        kwargs["tool_rag_top_k"] = tool_rag_top_k

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        **kwargs,
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "raw_response": raw_str[:500],
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
        "raw_response": raw.get("response", ""),
    }


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# --- Text generation (for demo chat layer, not tool calling) ---

def generate_local_text(query: str) -> dict:
    """Generate a text response locally via LFM2.5-1.2B on-device."""
    model = cactus_init(lfm_path)
    start = time.time()
    raw_str = cactus_complete(
        model,
        [
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": query},
        ],
        max_tokens=512,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)
    elapsed = (time.time() - start) * 1000
    try:
        raw = json.loads(raw_str)
        return {
            "response": raw.get("response", ""),
            "total_time_ms": raw.get("total_time_ms", elapsed),
            "confidence": raw.get("confidence", 0),
            "model": "LFM2.5-1.2B",
        }
    except json.JSONDecodeError:
        return {"response": raw_str, "total_time_ms": elapsed, "confidence": 0, "model": "LFM2.5-1.2B"}


def generate_cloud_text(query: str) -> dict:
    """Generate a text response via Gemini Cloud API."""
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    start = time.time()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[query],
    )
    elapsed = (time.time() - start) * 1000
    return {
        "response": response.text,
        "total_time_ms": elapsed,
        "model": "Gemini 2.5 Flash",
    }


# Heuristics for routing text queries to local vs cloud
COMPLEX_SIGNALS = re.compile(
    r'\b(explain|why|how does|compare|analyze|recommend|diagnose|summarize|'
    r'what should|pros and cons|difference between)\b', re.IGNORECASE
)


def should_route_text_to_cloud(query: str) -> bool:
    """Decide if a text query needs cloud LLM or can be handled locally."""
    # Long queries → cloud
    if len(query.split()) > 25:
        return True
    # Complex reasoning signals → cloud
    if COMPLEX_SIGNALS.search(query):
        return True
    return False


# --- Smart Routing Heuristics (for tool calling / benchmark) ---

# Patterns that signal multiple intents in a single query
MULTI_INTENT_PATTERNS = [
    r'\band\b',       # "set alarm and check weather"
    r'\bthen\b',      # "do X then Y"
    r'\balso\b',      # "also play music"
    r',\s*(?:and\b)?', # comma-separated actions
]
MULTI_INTENT_RE = re.compile('|'.join(MULTI_INTENT_PATTERNS), re.IGNORECASE)

# Action verbs that indicate separate tool calls
ACTION_VERBS = [
    'set', 'send', 'get', 'check', 'play', 'find', 'search', 'look',
    'remind', 'create', 'text', 'wake', 'start',
]
ACTION_VERB_RE = re.compile(
    r'\b(' + '|'.join(ACTION_VERBS) + r')\b', re.IGNORECASE
)


def _detect_multi_intent(user_msg):
    """Detect if a user message contains multiple tool-call intents."""
    # Count distinct action verbs
    verbs = set(m.group(1).lower() for m in ACTION_VERB_RE.finditer(user_msg))
    if len(verbs) >= 2 and MULTI_INTENT_RE.search(user_msg):
        return True
    return False


def _estimate_expected_calls(user_msg):
    """Estimate how many function calls the query expects."""
    verbs = set(m.group(1).lower() for m in ACTION_VERB_RE.finditer(user_msg))
    # If there's a conjunction and multiple verbs, likely multi-call
    if MULTI_INTENT_RE.search(user_msg) and len(verbs) >= 2:
        return len(verbs)
    return 1


def should_route_to_cloud(messages, tools, local_result):
    """Determine if we should fall back to cloud based on multiple signals."""
    user_msg = messages[-1]["content"]
    num_tools = len(tools)
    confidence = local_result.get("confidence", 0)
    predicted_calls = local_result.get("function_calls", [])

    # Signal 1: Multi-intent detection
    multi_intent = _detect_multi_intent(user_msg)
    expected_calls = _estimate_expected_calls(user_msg)

    # Signal 2: Predicted fewer calls than expected (strong cloud signal)
    if multi_intent and len(predicted_calls) < expected_calls:
        return True

    # Signal 3: Adaptive confidence threshold based on complexity
    if num_tools <= 1:
        # Easy: single tool — trust local almost always
        threshold = 0.5
    elif num_tools <= 3:
        # Medium: 2-3 tools
        threshold = 0.7
    else:
        # Hard: 4+ tools
        threshold = 0.8

    if confidence < threshold:
        return True

    return False


def _cloud_with_pii_protection(messages, tools, local_result):
    """Route to cloud with PII redaction/rehydration when DEMO_MODE is on."""
    vault = PIIVault()
    user_msg = messages[-1]["content"]

    if DEMO_MODE and has_pii_signals(user_msg):
        # Redact PII from the user message before sending to cloud
        redacted_msg, vault = redact(user_msg)
        redacted_messages = messages[:-1] + [
            {"role": messages[-1]["role"], "content": redacted_msg}
        ]
    else:
        redacted_messages = messages

    cloud = generate_cloud(redacted_messages, tools)
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local_result.get("confidence", 0)
    cloud["total_time_ms"] += local_result.get("total_time_ms", 0)

    # Rehydrate PII in function call arguments
    if vault.mappings:
        cloud["pii_redacted"] = True
        cloud["pii_vault"] = vault
        for call in cloud.get("function_calls", []):
            for key, val in call.get("arguments", {}).items():
                if isinstance(val, str):
                    call["arguments"][key] = rehydrate(val, vault)

    return cloud


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Smart hybrid inference: route to local or cloud based on heuristics."""
    user_msg = messages[-1]["content"]
    num_tools = len(tools)
    multi_intent = _detect_multi_intent(user_msg)

    # --- Fast path: single tool, no multi-intent → always local ---
    if num_tools == 1 and not multi_intent:
        local = generate_cactus(messages, tools)
        local["source"] = "on-device"
        return local

    # --- Medium path: multiple tools but single intent → local with RAG ---
    if not multi_intent:
        rag_k = min(num_tools, 3)  # narrow down tool selection
        local = generate_cactus(messages, tools, tool_rag_top_k=rag_k)

        if not should_route_to_cloud(messages, tools, local):
            local["source"] = "on-device"
            return local

        return _cloud_with_pii_protection(messages, tools, local)

    # --- Hard path: multi-intent detected → try local first, likely cloud ---
    local = generate_cactus(messages, tools)

    if not should_route_to_cloud(messages, tools, local):
        local["source"] = "on-device"
        return local

    return _cloud_with_pii_protection(messages, tools, local)


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
