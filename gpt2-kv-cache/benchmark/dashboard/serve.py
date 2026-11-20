"""Serve the live benchmark dashboard on this PC only.

Run from gpt2-kv-cache/: uv run python benchmark/dashboard/serve.py
Open: http://localhost:8765
"""

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8765


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/load.json"):
            body = json.dumps(dict(load_1min=round(os.getloadavg()[0], 2), cores=os.cpu_count())).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):  # keep the terminal quiet
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), partial(Handler, directory=str(HERE)))
    print(f"Dashboard: http://localhost:{PORT}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
