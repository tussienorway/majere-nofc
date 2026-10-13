"""The repo's own `<|role|>` training text format: single source of truth for training and serving.

`train/build_data.py` formats SFT examples with `fmt`; `nofc/serve.py` builds inference prompts with
`generation_prompt` from the same constants, so the two cannot drift. Standard library only (no torch needed).

Training text for one example:

    <|system|>\\n{system}\\n<|user|>\\n{text}\\n<|assistant|>\\n{text}\\n<|end|>

Tool calls are assistant text: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
"""
import json, re

from .evaluate import CALL_RE, PROMPT_TOOLS, THINK_RE

END = "<|end|>"
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
ROLES = ("system", "user", "assistant", "tool")
ROLE_TAG_RE = re.compile(r"<\|(?:system|user|assistant|tool)\|>")
TOOLS_MARKER = PROMPT_TOOLS.split("{tools}")[0].strip()


def role_tag(role):
    return f"<|{role}|>\n"


def body(system, turns):
    s = f"<|system|>\n{system}\n" if system else ""
    for role, text in turns:
        s += f"<|{role}|>\n{text}\n"
    return s


def fmt(system, turns):
    """Full training example text (what build_data.py writes to the `text` field)."""
    return body(system, turns) + END


def generation_prompt(system, turns):
    """Inference prompt: the training text up to and including the assistant tag of the turn to generate."""
    return body(system, turns) + role_tag("assistant")


def wrap_tool_call(obj):
    return TOOL_OPEN + json.dumps(obj, ensure_ascii=False) + TOOL_CLOSE


def format_tool_call(name, arguments):
    return wrap_tool_call({"name": name, "arguments": arguments})


def clean_completion(text):
    """Cut raw model output at END / the next role tag and drop <think> blocks."""
    end = text.find(END)
    if end >= 0: text = text[:end]
    tag = ROLE_TAG_RE.search(text)
    if tag: text = text[:tag.start()]
    return THINK_RE.sub("", text).strip()


def parse_completion(text):
    """Model output -> (content, calls): extracts valid tool calls from the cleaned text.

    Malformed `<tool_call>` blocks are left in `content` so they are never silently scored as a no-call success.
    """
    text = clean_completion(text)
    calls = []

    def take(m):
        try:
            j = json.loads(m.group(1))
        except json.JSONDecodeError:
            return m.group(0)
        if not isinstance(j, dict) or not isinstance(j.get("name"), str):
            return m.group(0)
        args = j.get("arguments") or {}
        if isinstance(args, str):
            try: args = json.loads(args)
            except json.JSONDecodeError: return m.group(0)
        calls.append({"name": j["name"], "arguments": args})
        return ""

    content = CALL_RE.sub(take, text).strip()
    return content, calls


def _text(content):
    if content is None: return ""
    if isinstance(content, str): return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type", "text") == "text")


def _tool_calls_text(tool_calls):
    out = []
    for tc in tool_calls:
        fn = tc.get("function", tc)
        args = fn.get("arguments")
        if isinstance(args, str):
            try: args = json.loads(args)
            except json.JSONDecodeError: pass
        out.append(format_tool_call(fn.get("name"), args if args is not None else {}))
    return "\n".join(out)


def with_tools(system, tools):
    """Append the tool schemas exactly like `evaluate --tool-mode prompt` does (PROMPT_TOOLS from evaluate.py)."""
    if not tools or TOOLS_MARKER in (system or ""): return system
    fns = [t["function"] if "function" in t else t for t in tools]
    block = PROMPT_TOOLS.format(tools=json.dumps(fns, ensure_ascii=False))
    return system + block if system else block.strip()


def messages_to_turns(messages, tools=None):
    """OpenAI-style messages (+ optional native `tools`) -> (system, turns) for `generation_prompt`."""
    system = "\n\n".join(_text(m.get("content")) for m in messages if m["role"] in ("system", "developer")) or None
    turns = []
    for m in messages:
        role = m["role"]
        if role in ("system", "developer"): continue
        if role not in ROLES: raise ValueError(f"unsupported role {role!r}")
        text = _text(m.get("content"))
        if role == "assistant" and m.get("tool_calls"):
            text = (text + "\n" if text else "") + _tool_calls_text(m["tool_calls"])
        turns.append((role, text))
    return with_tools(system, tools), turns


def build_prompt(messages, tools=None):
    return generation_prompt(*messages_to_turns(messages, tools))
