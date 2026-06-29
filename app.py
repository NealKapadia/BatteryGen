"""Local molecule-design server (zero external deps — Python stdlib only).

Thin wrapper around design.DesignEngine. For production use server.py (FastAPI) +
Docker; this file is for quick local use during development.

  python molvae/app.py                 # CPU by default; opens http://localhost:8000
  python molvae/app.py --device cuda   # after training finishes
"""
from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"
ENGINE = None
ARGS = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, json.dumps({"ok": True, "device": str(ENGINE.device),
                                        "electrolyte": ENGINE.elec is not None}))
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._send(400, json.dumps({"ok": False, "error": "bad json"}))
        try:
            if self.path == "/design":
                out = ENGINE.design(req.get("prompt", ""), req.get("sliders", {}))
            elif self.path == "/refine":
                out = ENGINE.design(req.get("prompt", ""), req.get("sliders", {}), refine=True)
            elif self.path == "/explain":
                out = ENGINE.explain()
            elif self.path == "/finetune":
                out = ENGINE.finetune(req.get("smiles", []))
            else:
                out = {"ok": False, "error": "unknown endpoint"}
        except Exception as e:
            out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self._send(200, json.dumps(out))


def main():
    global ENGINE, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-browser", action="store_true")
    ARGS = ap.parse_args()

    import design
    print(f"Loading model on {ARGS.device} ...")
    ENGINE = design.DesignEngine(ckpt=ARGS.ckpt, device=ARGS.device)
    print(f"Ready. Open http://localhost:{ARGS.port}")
    if not ARGS.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{ARGS.port}")).start()
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
