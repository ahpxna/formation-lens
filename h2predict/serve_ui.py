"""Static server for ui/ with HTTP Range support: browsers need 206 partial responses to seek inside a video,
which `python -m http.server` does not provide."""
import argparse
import errno
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        if not getattr(self, "_partial", False):
            self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")  # always revalidate: a stale index.html broke a demo reload
        super().end_headers()

    def send_head(self):
        self._partial, self._remaining = False, None
        m = RANGE.match((self.headers.get("Range") or "").strip())
        path = self.translate_path(self.path)
        if not m or os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        size = os.fstat(f.fileno()).st_size
        if m.group(1) == "":
            start, end = max(0, size - int(m.group(2) or 0)), size - 1
        else:
            start, end = int(m.group(1)), int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            f.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        self._partial, self._remaining = True, end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._remaining))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        f.seek(start)
        return f

    def copyfile(self, source, outputfile):
        if self._remaining is None:
            return super().copyfile(source, outputfile)
        while self._remaining > 0:
            chunk = source.read(min(1 << 16, self._remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            self._remaining -= len(chunk)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="ui")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))  # PORT: set by preview tools
    a = ap.parse_args(argv)
    try:
        server = ThreadingHTTPServer((a.host, a.port), partial(RangeHandler, directory=a.dir))
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        ap.error(f"cổng {a.port} đang được dùng (có thể UI đã chạy sẵn: mở http://{a.host}:{a.port}); "
                 f"hoặc chọn cổng khác với --port {a.port + 1}")
    print(f"serving {a.dir}/ at http://{a.host}:{a.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
