"""Dashboard server for Rescue City.

Binds DUAL-STACK (IPv6 + IPv4) so BOTH http://localhost:8000 and
http://127.0.0.1:8000 work. The default `python -m http.server --bind 127.0.0.1`
only listens on IPv4, but modern browsers resolve `localhost` to IPv6 (::1) first
and then show a blank "can't connect" page — that was the whole "nothing loads"
problem. This server avoids it.

Run from the webots/ folder:  python serve.py   (or just run_dashboard.bat)
Then open:  http://localhost:8000/dashboard/   (127.0.0.1 also works)
"""
import http.server
import json
import os
import socket
import socketserver
import subprocess

PORT = 8000
WEBOTS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(WEBOTS_DIR)   # serve the webots/ folder

# Webots launcher (the .exe, NOT webots-bin.exe — the wrapper sets up the DLL env).
WEBOTS_EXE = os.environ.get("WEBOTS_EXE", r"E:\Webots\msys64\mingw64\bin\webots.exe")
WORLD = os.path.join(WEBOTS_DIR, "worlds", "rescue_city_mesh.wbt")
WBPROJ = os.path.join(WEBOTS_DIR, "worlds", ".rescue_city_mesh.wbproj")
# mission state the dashboard reads — cleared on reset so the panels start blank
STATE_FILES = ["mission.json", "mission_log.json", "backup_request.json"]


def _kill_webots():
    """Force-kill every Webots process so we never pile up instances (the pile-up
    crashed the laptop once). Returns True if anything was running."""
    out = subprocess.run(
        ["taskkill", "/F", "/IM", "webots-bin.exe", "/IM", "webotsw.exe", "/IM", "webots.exe"],
        capture_output=True, text=True)
    return "SUCCESS" in (out.stdout or "")


def _webots_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq webots-bin.exe"],
                         capture_output=True, text=True)
    return "webots-bin.exe" in (out.stdout or "")


def _clear_state():
    """Wipe the dashboard's mission files + live frames so a reset starts clean."""
    import glob
    # the fixed state files + every per-drone log/position file (so old logs don't linger)
    targets = list(STATE_FILES)
    targets += [os.path.basename(p) for p in glob.glob(os.path.join(WEBOTS_DIR, "mission_log_*.json"))]
    targets += [os.path.basename(p) for p in glob.glob(os.path.join(WEBOTS_DIR, "live_pos_*.json"))]
    targets += ["mission_error.log"]
    for fn in targets:
        try:
            os.remove(os.path.join(WEBOTS_DIR, fn))
        except OSError:
            pass
    frames = os.path.join(WEBOTS_DIR, "frames")
    if os.path.isdir(frames):
        for fn in os.listdir(frames):
            if fn.endswith(".png") or fn.endswith(".json"):
                try:
                    os.remove(os.path.join(frames, fn))
                except OSError:
                    pass


def _launch_webots():
    """Launch a SINGLE Webots instance (kills any existing first). Deletes the
    .wbproj perspective so the 14 stale camera overlays can't freeze the GUI."""
    _kill_webots()
    try:
        os.remove(WBPROJ)          # avoid the camera-overlay storm that hangs the GUI
    except OSError:
        pass
    if not os.path.exists(WEBOTS_EXE):
        return False, f"Webots not found at {WEBOTS_EXE} (set WEBOTS_EXE env var)"
    # Run BACKGROUND/headless so the heavy 3D GUI never bogs down or freezes:
    #   --minimize     : no main window shown
    #   --no-rendering : skip main-view 3D rendering (the drone cameras still render
    #                    offscreen, so the dashboard keeps getting live frames)
    #   --batch        : never pop a dialog
    # The dashboard is the display; Webots just simulates in the background.
    subprocess.Popen([WEBOTS_EXE, "--batch", "--minimize", "--no-rendering",
                      "--mode=realtime", WORLD],
                     cwd=WEBOTS_DIR, creationflags=subprocess.CREATE_NO_WINDOW)
    return True, "launched"


class DualStackServer(socketserver.ThreadingTCPServer):
    """Threaded so concurrent image/JSON polls never block each other; dual-stack
    so localhost (IPv6) and 127.0.0.1 (IPv4) both connect."""
    allow_reuse_address = True
    daemon_threads = True
    address_family = socket.AF_INET6

    def server_bind(self):
        # accept IPv4 connections on the IPv6 socket too (127.0.0.1 + ::1)
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            pass
        super().server_bind()


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # never let the browser cache live frames / json / the dashboard html
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        super().end_headers()

    def log_message(self, *a):
        pass   # quiet

    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/api/status":
            return self._send_json({"running": _webots_running()})
        return super().do_GET()

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/api/run":
            ok, msg = _launch_webots()
            return self._send_json({"ok": ok, "msg": msg, "running": ok})
        if path == "/api/reset":
            _kill_webots()
            _clear_state()
            ok, msg = _launch_webots()
            return self._send_json({"ok": ok, "msg": "reset+" + msg, "running": ok})
        if path == "/api/stop":
            was = _kill_webots()
            return self._send_json({"ok": True, "msg": "stopped" if was else "not running",
                                    "running": False})
        self._send_json({"error": "unknown endpoint"}, code=404)


if __name__ == "__main__":
    try:
        httpd = DualStackServer(("", PORT), Handler)
    except OSError:
        # no IPv6 available -> fall back to plain IPv4 on all interfaces
        socketserver.ThreadingTCPServer.address_family = socket.AF_INET
        httpd = socketserver.ThreadingTCPServer(("0.0.0.0", PORT), Handler)
        httpd.allow_reuse_address = True
    print("=" * 60)
    print("  Rescue City dashboard is live. Open either of these:")
    print(f"    http://localhost:8000/dashboard/")
    print(f"    http://127.0.0.1:8000/dashboard/")
    print("  (leave this window open during the demo; Ctrl+C to stop)")
    print("=" * 60, flush=True)
    httpd.serve_forever()
