"""Minimal OpenAI-compatible server for a QLoRA adapter trained with train/qlora_sft.py (greedy decoding).

The base model is loaded in 4-bit NF4 exactly as in training with the LoRA adapter on top.
Prompts are built by nofc.chatfmt from the same constants as train/build_data.py, not by Qwen3's chat template.

  python -m nofc.serve --base qwen3-8b --adapter runs/qwen3-8b-r16-s0 --port 8000
  python -m nofc.serve --base qwen3-8b --port 8001          # same format, no adapter: control for the comparison

Needs the train/requirements.txt packages. The request/response handling below imports nothing heavy.
"""
import argparse, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .chatfmt import END, build_prompt, clean_completion, parse_completion
from .trainutil import BASE_MODELS, BNB_4BIT

DEFAULT_MAX_NEW_TOKENS = 512


def handle_chat(body, generate, model_name):
    """OpenAI chat.completions request dict -> response dict. `generate(prompt, max_new_tokens) -> str` is injected."""
    prompt = build_prompt(body["messages"], body.get("tools"))
    max_new = min(int(body.get("max_tokens") or DEFAULT_MAX_NEW_TOKENS), DEFAULT_MAX_NEW_TOKENS * 4)
    raw = generate(prompt, max_new)
    content, calls = parse_completion(raw)
    msg = {"role": "assistant", "content": content or None}
    if calls and body.get("tools"):
        msg["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                              "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                             for i, c in enumerate(calls)]
    elif calls:
        # no tools declared (evaluate --tool-mode prompt): keep the <tool_call> text for the client to parse
        msg["content"] = clean_completion(raw)
    return {"id": f"chatcmpl-{int(time.time() * 1000)}", "object": "chat.completion", "created": int(time.time()),
            "model": model_name,
            "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop"}]}


class HFGenerator:
    """transformers + peft backend (loaded lazily so the module imports without torch)."""

    def __init__(self, base, adapter):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self.torch = torch
        name = BASE_MODELS[base]
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        bnb = BitsAndBytesConfig(**BNB_4BIT, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb, device_map={"": 0},
                                                     dtype=torch.bfloat16, attn_implementation="sdpa")
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        self.model = model.eval()
        self.lock = threading.Lock()

    def __call__(self, prompt, max_new_tokens):
        enc = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with self.lock, self.torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, stop_strings=[END],
                                      tokenizer=self.tok, pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def make_server(generate, model_name, host="127.0.0.1", port=8000):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "not found"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                resp = handle_chat(body, generate, model_name)
            except (ValueError, KeyError, TypeError) as e:
                return self._send(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
            except Exception as e:
                return self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            self._send(200, resp)

        def log_message(self, *a): pass

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", choices=BASE_MODELS, required=True)
    p.add_argument("--adapter", default=None, help="adapter dir written by train/qlora_sft.py (omit to serve the plain base model)")
    p.add_argument("--served-model-name", default=None, help="id reported to clients (default: <base>-adapter or <base>-base)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    name = a.served_model_name or f"{a.base}-{'adapter' if a.adapter else 'base'}"
    gen = HFGenerator(a.base, a.adapter)
    srv = make_server(gen, name, a.host, a.port)
    print(f"serving {name} at http://{a.host}:{a.port}/v1 (greedy, 4-bit NF4 base, adapter={a.adapter})", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass


if __name__ == "__main__":
    main()
