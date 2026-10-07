"""Loopback-only web UI and a separate, long-lived simulation process."""
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time
import webbrowser
import yaml
from .live_controls import Controls, validate_patch
from .live_worker import simulation_worker


class LiveServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, config, output):
        super().__init__(address, Handler)
        context = mp.get_context("spawn")
        self.commands = context.Queue(maxsize=128)
        self.results = context.Queue(maxsize=2)
        self.stop_worker = context.Event()
        self.state_lock = threading.Lock()
        self.state = {"status": "loading", "message": "Starting simulation process..."}
        self.frame = b""
        self.inner_frame = b""
        self.frame_version = 0
        self.sequence = 0
        self.geometry_path = Path(output) / "brain_geometry.json"
        self.desired = Controls(source_x=config["body"]["odor_source_mm"][0],
                                source_y=config["body"]["odor_source_mm"][1])
        self.worker = context.Process(target=simulation_worker,
                                      args=(config, self.commands, self.results, self.stop_worker, str(output)),
                                      name="VirtualFlySimulation")
        self.worker.start()
        threading.Thread(target=self.collect, daemon=True).start()

    def collect(self):
        while not self.stop_worker.is_set():
            try:
                packet = self.results.get(timeout=.25)
            except queue.Empty:
                if not self.worker.is_alive():
                    with self.state_lock:
                        if self.state.get("status") not in ("stopped", "error"):
                            self.state.update(status="error", message="Simulation process exited. Check the launcher log.")
                    break
                continue
            with self.state_lock:
                self.state = packet["state"]
                if packet.get("frame") is not None:
                    self.frame = packet["frame"]
                    self.frame_version += 1
                if packet.get("inner_frame") is not None:
                    self.inner_frame = packet["inner_frame"]

    def command(self, request):
        if not isinstance(request, dict):
            raise ValueError("Expected a JSON object")
        if set(request) == {"patch"}:
            patch = validate_patch(request["patch"])
            item = {"patch": patch}
        elif request == {"action": "step"}:
            item = {"action": "step"}
        else:
            raise ValueError("Expected a control patch or step action")
        with self.state_lock:
            if self.state.get("status") in ("error", "stopped"):
                raise ValueError("Session is not running; restart the launcher")
            seq = self.sequence + 1
            self.commands.put_nowait({"seq": seq, **item})
            self.sequence = seq
            if "patch" in item:
                self.desired.update(item["patch"])
            else:
                self.desired.paused = True
        return seq

    def finish(self):
        self.stop_worker.set()
        self.worker.join(timeout=10)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=3)
        self.server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def allowed_host(self):
        port = self.server.server_address[1]
        return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

    def send(self, body, content_type="application/json", status=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if not self.allowed_host():
            return self.send({"error": "Invalid host"}, status=403)
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self.send(Path(__file__).with_name("live.html").read_bytes(), "text/html; charset=utf-8")
        assets = {
            "/static/brain-view.js": "static/brain-view.js",
            "/static/three/three.module.js": "static/three/three.module.js",
            "/static/three/three.core.js": "static/three/three.core.js",
        }
        if path in assets:
            return self.send((Path(__file__).parent / assets[path]).read_bytes(), "text/javascript; charset=utf-8")
        if path == "/api/brain_geometry":
            if not self.server.geometry_path.exists():
                return self.send({"loading": True}, status=202)
            return self.send(self.server.geometry_path.read_bytes())
        if path == "/api/state":
            with self.server.state_lock:
                value = {**self.server.state, "accepted_seq": self.server.sequence,
                         "frame_version": self.server.frame_version}
            return self.send(value)
        if path in ("/frame.jpg", "/inner_frame.jpg"):
            with self.server.state_lock:
                frame = self.server.inner_frame if path == "/inner_frame.jpg" else self.server.frame
            return self.send(frame, "image/jpeg", status=200 if frame else 204)
        self.send({"error": "Not found"}, status=404)

    def do_POST(self):
        port = self.server.server_address[1]
        origin = self.headers.get("Origin")
        if not self.allowed_host() or origin not in (None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            return self.send({"error": "Local same-origin requests only"}, status=403)
        if self.headers.get_content_type() != "application/json":
            return self.send({"error": "JSON required"}, status=415)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(size))
            if self.path == "/api/stop" and request == {"stop": True}:
                self.send({"stopped": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/api/control":
                return self.send({"error": "Not found"}, status=404)
            seq = self.server.command(request)
            self.send({"accepted_seq": seq})
        except queue.Full:
            self.send({"error": "Command queue is full; wait for the current step"}, status=429)
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send({"error": str(error)}, status=400)


def live_run(args, config):
    nested_path = Path("configs/nested.yaml")
    if "nested" not in config and nested_path.exists():
        config["nested"] = yaml.safe_load(nested_path.read_text(encoding="utf-8"))
    # Every launch gets a new directory; continuous CSV logs are flushed each tick.
    output = Path(args.output) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    server = LiveServer(("127.0.0.1", args.port), config, output)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Interactive fly: {url}\nSession logs: {output.resolve()}\nClose with End session or Ctrl+C.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.finish()
