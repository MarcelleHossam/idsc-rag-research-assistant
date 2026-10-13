"""
Web server for the IDSC research-assistant GUI.

Loads the whole pipeline in idsc_project.py once at startup (same thing that
happens when you run "python idsc_project.py" directly - it reads the cached
transcripts/PDFs, opens the Chroma db, and gets ready to answer). Then exposes
two small JSON endpoints the page in web/templates/index.html calls:

    POST /api/ask       {"question": "..."}  -> {"answer", "sources", "scope"}
    GET  /api/sources    -> list of every video/publication in the index

Run with:
    python server.py
then open http://127.0.0.1:5000 in your browser.
"""
import os
from flask import Flask, jsonify, render_template, request

import idsc_project as rag  # runs the whole setup (cache load + chroma) once

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "web", "templates"),
    static_folder=os.path.join(BASE_DIR, "web", "static"),
    static_url_path="/assets",
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/sources")
def api_sources():
    return jsonify(rag.SOURCES)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    body = request.get_json(force=True, silent=True) or {}
    question = (body.get("question") or "").strip()
    if not question:
        return jsonify({"error": "الرجاء كتابة سؤال."}), 400

    try:
        answer, results, scope = rag.ask(question)
    except Exception as e:
        return jsonify({"error": f"حدث خطأ أثناء البحث: {e}"}), 500

    sources = [
        {
            "title": r["meta"]["title"],
            "locator": r["meta"]["locator"],
            "url": r["meta"].get("url", ""),
            "type": r["meta"]["source_type"],
        }
        for r in results
    ]
    return jsonify({"answer": answer, "sources": sources, "scope": scope})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
