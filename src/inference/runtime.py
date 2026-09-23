"""Small offline Transformers HTTP runtime for locally downloaded causal models."""
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer


def load_generator():
    if os.environ.get("VLLM_ENABLED", "false").lower() == "true":
        raise ValueError("This runtime uses Transformers, not vLLM")
    if os.environ.get("QUANTIZATION", "none") != "none":
        raise ValueError("This runtime does not support quantization overrides")
    max_sequence = int(os.environ.get("MAX_SEQ_LEN", "2048"))
    if max_sequence < 2:
        raise ValueError("MAX_SEQ_LEN must be at least two")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = os.environ.get("MODEL_PATH", "/model")
    device = os.environ.get("DEVICE", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("DEVICE must be cpu, cuda, or auto")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    model.to(device).eval()
    context = min(max_sequence, getattr(model.config, "max_position_embeddings", max_sequence))
    def generate(prompt, max_new_tokens):
        max_new_tokens = min(max_new_tokens, context - 1)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context - max_new_tokens).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                    pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate


def handler_for(generate):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ("/health", "/healthz"):
                self.reply(200, {"status": "ready"})
            else:
                self.reply(404, {"error": "Not found"})

        def do_POST(self):
            if self.path != "/generate":
                return self.reply(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 65536:
                    raise ValueError("Request must be between 1 and 65536 bytes")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                prompt = body.get("prompt")
                tokens = body.get("max_new_tokens", 64)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("prompt must be a nonempty string")
                if type(tokens) is not int or not 1 <= tokens <= 512:
                    raise ValueError("max_new_tokens must be an integer between 1 and 512")
            except (ValueError, UnicodeError) as exc:
                return self.reply(400, {"error": str(exc)})
            try:
                self.reply(200, {"text": generate(prompt, tokens)})
            except Exception:
                self.reply(500, {"error": "Generation failed; check runtime logs"})
    return Handler


def main():
    generator = load_generator()  # Readiness is available only after loading succeeds.
    server = HTTPServer((os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", "8000"))), handler_for(generator))
    server.serve_forever()


if __name__ == "__main__":
    main()
