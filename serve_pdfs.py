"""
Serves the publication PDFs at the exact URL the citations use:

    http://localhost:8501/app/static/64.pdf#page=13

That is the path Streamlit's static serving uses, so run this only when the
app is not already running - whichever process owns the port serves the file,
and the links in the answers stay the same either way.

    python serve_pdfs.py            8501
    python serve_pdfs.py 9000       another port; then set IDSC_APP_URL to
                                    http://localhost:9000 and import the
                                    pipeline once to relabel the links
"""
import os
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PREFIX = "/app/static/"


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        name = path.split("?", 1)[0].split("#", 1)[0]
        if name.startswith(PREFIX):
            name = name[len(PREFIX):]
        return os.path.join(STATIC_DIR, os.path.basename(name))

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write("%s\n" % (fmt % args))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    if not os.path.isdir(STATIC_DIR) or not os.listdir(STATIC_DIR):
        sys.exit('static/ is empty - run "python idsc project.py" once to fill it')
    served = sorted(f for f in os.listdir(STATIC_DIR) if f.endswith(".pdf"))
    print(f"serving {len(served)} PDFs on http://localhost:{port}{PREFIX}")
    for f in served:
        print(f"  http://localhost:{port}{PREFIX}{f}")
    ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=STATIC_DIR)).serve_forever()


if __name__ == "__main__":
    main()
