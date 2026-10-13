import os, json, sys, re, glob
import chromadb
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
from dotenv import load_dotenv

for _stream in (sys.stdout, sys.stdin):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

STATIC_DIR = os.path.join(BASE_DIR, "static")
APP_URL = os.environ.get("IDSC_APP_URL", "").rstrip("/")

load_dotenv(os.path.join(BASE_DIR, ".env"))
FIREWORKS_API_KEY = os.environ["FIREWORKS_API_KEY"]
EMBED_MODEL = "fireworks/qwen3-embedding-8b"
CHAT_MODEL  = "accounts/fireworks/models/deepseek-v4-pro"

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=FIREWORKS_API_KEY,
)

VIDEOS = [
    ("uvrOHj1tO1c", "الذكاء الاصطناعي والاقتصاد الرقمي في مصر"),
    ("kqkOsZ_D1qs", "المنصة الاقتصادية — التضخم"),
    ("VzHMMvRQdrs", "مشروعات المنطقة الاقتصادية لقناة السويس"),
    ("U4P_3nmQhXo", "مبادرة ابدأ — دعم الصناعة المصرية"),
    ("tEP0gC2j2As", "جهاز EZVent"),
    ("XeZiSS1_PcI", "القاهرة في صدارة المدن الإفريقية الأكثر تأثيرًا"),
    ("jVleeTgWeYs", "الصحة الرقمية"),
    ("yoff3yQM780", "استثمارات الثروة السمكية"),
    ("_qZ6SQLPxt0", "دور الهيئة العامة للرقابة المالية في التحول الرقمي"),
    ("mOnaDiBL2-Y", "البنية التحتية في مصر"),
]

PUB_PAGE = "https://www.idsc.gov.eg/Publication/details/{}"

PUBLICATIONS = [
    {"key": "67", "pub": "11507", "pdf_url": "", "title": "مستقبل العملات الرقمية"},
    {"key": "66", "pub": "11500", "pdf_url": "", "title": "تغيرات في السياسات التجارية"},
    {"key": "64", "pub": "11457", "pdf_url": "", "title": "آليات تعزيز الإيرادات العامة"},
    {"key": "63", "pub": "11438", "pdf_url": "", "title": "الذكاء الاصطناعي والنمو الاقتصادي"},
    {"key": "59", "pub": "11366", "pdf_url": "", "title": "الاستقرار المالي العالمي"},
    {"key": "9",  "pub": "9041",  "pdf_url": "", "title": "حرب غزة وآفاق التسوية"},
    {"key": "11", "pub": "12517", "pdf_url": "", "title": "التحولات الجيوسياسية ومستقبل الشرق الأوسط"},
]

for _p in PUBLICATIONS:
    _p["link"] = f"{APP_URL}/app/static/{_p['key']}.pdf"
    _p["page"] = PUB_PAGE.format(_p["pub"]) if _p["pub"] else ""

LANGS = ["ar", "en"]
MAX_CHARS = 400
OVERLAP_SEGS = 2
PDF_MAX_CHARS = 900
PDF_OVERLAP = 150
REFRESH = False
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

def hhmmss(sec):
    s = int(sec)
    return f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

def norm_ar(s):
    """Fold Arabic spelling variants so title matching survives real questions."""
    s = s.lower().translate(_ARABIC_DIGITS)
    s = re.sub(r"[ً-ْـ]", "", s)
    for a in "أإآ":
        s = s.replace(a, "ا")
    return s.replace("ة", "ه").replace("ى", "ي")

def chunk_id(c):
    return f"{c['source_id']}-{c['index']:04d}"

def meta_of(c):
    return {
        "source_type": c["source_type"],
        "source_id":   c["source_id"],
        "title":       c["title"],
        "locator":     c["locator"],
        "url":         c["url"],
        "index":       c["index"],
    }


def make_chunk(segs, vid, title, index):
    s = int(segs[0]["start"])
    return {
        "text": " ".join(x["text"] for x in segs),
        "source_type": "video", "source_id": vid, "title": title, "index": index,
        "locator": hhmmss(s),
        "url": f"https://youtube.com/watch?v={vid}&t={s}s",
    }

def normalize_chunk(c):
    """Accept both the old (video_id/timestamp) and new cache formats."""
    if "source_id" in c:
        return c
    return {
        "text": c["text"], "source_type": "video", "source_id": c["video_id"],
        "title": c["title"], "index": c["index"],
        "locator": c.get("timestamp", ""), "url": c["url"],
    }

def load_video(vid, title):
    cache_path = os.path.join(DATA_DIR, f"{vid}.json")

    if not REFRESH and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            chunks = [normalize_chunk(c) for c in json.load(f)]
        for c in chunks:
            c["title"] = title
        print(f"[cache] {vid}  {title[:30]:30}  {len(chunks)} chunks")
        return chunks

    try:
        segments = api.fetch(vid, languages=LANGS).to_raw_data()
    except Exception as e:
        print(f"[skip]  {vid}  {title[:30]}  {type(e).__name__}")
        return []

    chunks, buf = [], []
    pending = False
    for seg in segments:
        buf.append(seg)
        pending = True
        if sum(len(x["text"]) for x in buf) >= MAX_CHARS:
            chunks.append(make_chunk(buf, vid, title, len(chunks)))
            buf = buf[-OVERLAP_SEGS:] if OVERLAP_SEGS else []
            pending = False
    if pending:
        chunks.append(make_chunk(buf, vid, title, len(chunks)))

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[fetch] {vid}  {title[:30]:30}  {len(chunks)} chunks")
    return chunks


def extract_pages(path):
    """[(page_number, text)].

    The IDSC PDFs embed subsetted Tajawal faces with a corrupt ToUnicode map, so
    a plain text extraction returns mojibake. demangle rebuilds the mapping from
    the glyph outlines; fall back to plain extraction if it is unavailable.
    """
    try:
        import demangle
        return list(demangle.doc_pages(path))
    except Exception as e:
        print(f"[warn]  demangle unavailable ({type(e).__name__}), reading as-is")
    try:
        import pymupdf
        with pymupdf.open(path) as doc:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    except ImportError:
        from pypdf import PdfReader
        return [(i + 1, p.extract_text() or "")
                for i, p in enumerate(PdfReader(path).pages)]

def make_pdf_chunk(text, pid, title, index, page, page_url):
    return {
        "text": text,
        "source_type": "pdf", "source_id": f"pub{pid}", "title": title, "index": index,
        "locator": f"ص {page}",
        "url": f"{page_url}#page={page}" if page_url else "",
    }

def chunk_pdf(path, pid, title, page_url=""):
    chunks = []
    for page, raw in extract_pages(path):
        text = re.sub(r"[ \t ]+", " ", raw)
        text = re.sub(r"\n{2,}", "\n", text).strip()
        if len(text) < 40:
            continue
        start = 0
        while start < len(text):
            piece = text[start:start + PDF_MAX_CHARS].strip()
            if len(piece) >= 40:
                chunks.append(
                    make_pdf_chunk(piece, pid, title, len(chunks), page, page_url))
            start += max(PDF_MAX_CHARS - PDF_OVERLAP, 1)
    return chunks

def publish_static(path, key):
    """Copy the PDF under static/<key>.pdf so the app can serve it."""
    import shutil
    os.makedirs(STATIC_DIR, exist_ok=True)
    dest = os.path.join(STATIC_DIR, f"{key}.pdf")
    if not os.path.exists(dest) or os.path.getsize(dest) != os.path.getsize(path):
        shutil.copy2(path, dest)
    return dest

def find_pdf(title):
    """pdfs/<title>.pdf, or any file starting with the title."""
    exact = os.path.join(PDF_DIR, f"{title}.pdf")
    if os.path.exists(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(PDF_DIR, f"{title}*.pdf")))
    return hits[0] if hits else None

def download_pdf(pid, url):
    import requests
    dest = os.path.join(PDF_DIR, f"{pid}.pdf")
    r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError("the link did not return a PDF")
    with open(dest, "wb") as f:
        f.write(r.content)
    return dest

def retitle(chunks, title, page_url):
    """Renaming a source only changes its label, never its text, so relabel the
    cached chunks in place instead of re-reading and re-embedding the PDF."""
    for c in chunks:
        c["title"] = title
        page = re.match(r"ص\s*(\d+)", c.get("locator", ""))
        c["url"] = f"{page_url}#page={page.group(1)}" if page_url and page else ""
    return chunks

def load_publication(pub):
    pid, title = pub["key"], pub["title"]
    cache_path = os.path.join(DATA_DIR, f"pub{pid}.json")

    if not REFRESH and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"[cache] pub{pid}  {title[:34]:34}  {len(chunks)} chunks")
        local = find_pdf(title)
        if local:
            publish_static(local, pid)
        return retitle(chunks, title, pub.get("link", ""))

    path = find_pdf(title)
    if not path and pub.get("pdf_url"):
        try:
            path = download_pdf(pid, pub["pdf_url"])
            print(f"[get]   pub{pid}  downloaded")
        except Exception as e:
            print(f"[skip]  pub{pid}  {title[:30]}  download failed: {type(e).__name__}")
            return []
    if not path:
        print(f"[skip]  pub{pid}  {title[:34]:34}  no PDF - save it as pdfs/{title}.pdf")
        return []

    publish_static(path, pid)
    try:
        chunks = chunk_pdf(path, pid, title, pub.get("link", ""))
    except Exception as e:
        print(f"[skip]  pub{pid}  {title[:30]}  {type(e).__name__}: {e}")
        return []
    if not chunks:
        print(f"[skip]  pub{pid}  {title[:30]:30}  no extractable text (scanned?)")
        return []

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[read]  pub{pid}  {title[:30]:30}  {len(chunks)} chunks")
    return chunks

def load_loose_pdfs(known):
    """Any other PDF dropped into pdfs/ joins the index under its file name."""
    out = []
    for path in sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf"))):
        name = os.path.splitext(os.path.basename(path))[0]
        if any(name.startswith(pid) for pid in known):
            continue
        cache_path = os.path.join(DATA_DIR, f"file-{name}.json")
        if not REFRESH and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                chunks = json.load(f)
        else:
            try:
                chunks = chunk_pdf(path, name, name)
            except Exception as e:
                print(f"[skip]  {name}  {type(e).__name__}")
                continue
            for c in chunks:
                c["source_id"] = f"file-{name}"
                c["url"] = ""
            if not chunks:
                continue
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
        print(f"[read]  {name[:34]:34}  {len(chunks)} chunks")
        out.extend(chunks)
    return out


api = YouTubeTranscriptApi()
all_chunks = []

for vid, title in VIDEOS:
    all_chunks.extend(load_video(vid, title))
for pub in PUBLICATIONS:
    all_chunks.extend(load_publication(pub))
all_chunks.extend(load_loose_pdfs({p["title"] for p in PUBLICATIONS}))

SOURCES = []
_seen = set()
for c in all_chunks:
    if c["source_id"] not in _seen:
        _seen.add(c["source_id"])
        SOURCES.append({
            "source_id": c["source_id"], "title": c["title"],
            "source_type": c["source_type"],
            "url": c["url"].split("&t=")[0].split("#page=")[0],
        })

with open(os.path.join(DATA_DIR, "all_chunks.json"), "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

db = chromadb.PersistentClient(path=CHROMA_DIR)
col = db.get_or_create_collection("idsc")

stored = col.get(include=["metadatas", "documents"])

legacy_ids, legacy_meta = [], []
for cid, meta in zip(stored["ids"], stored["metadatas"]):
    if "source_type" not in meta:
        legacy_ids.append(cid)
        legacy_meta.append({
            "source_type": "video", "source_id": meta.get("video_id", ""),
            "title": meta.get("title", ""), "locator": meta.get("timestamp", ""),
            "url": meta.get("url", ""), "index": int(cid.rsplit("-", 1)[-1]),
        })
if legacy_ids:
    col.update(ids=legacy_ids, metadatas=legacy_meta)
    position = {cid: i for i, cid in enumerate(stored["ids"])}
    for cid, meta in zip(legacy_ids, legacy_meta):
        stored["metadatas"][position[cid]] = meta
    print(f"relabelled {len(legacy_ids)} chunks to the new source schema")

wanted = {c["source_id"] for c in all_chunks}
loaded = wanted
current = {chunk_id(c): c["text"] for c in all_chunks}

stale = [
    cid for cid, meta, doc in zip(stored["ids"], stored["metadatas"], stored["documents"])
    if meta.get("source_id") not in wanted
    or (meta.get("source_id") in loaded and current.get(cid) != doc)
]
if stale:
    col.delete(ids=stale)
    print(f"removed {len(stale)} outdated chunks")

desired = {chunk_id(c): meta_of(c) for c in all_chunks}
stored = col.get(include=["metadatas"])
fix_ids, fix_meta = [], []
for cid, meta in zip(stored["ids"], stored["metadatas"]):
    want = desired.get(cid)
    if want and any(meta.get(k) != v for k, v in want.items()):
        fix_ids.append(cid)
        fix_meta.append(want)
if fix_ids:
    col.update(ids=fix_ids, metadatas=fix_meta)
    print(f"refreshed metadata on {len(fix_ids)} chunks")

have = set(col.get(include=[])["ids"])
todo = [c for c in all_chunks if chunk_id(c) not in have]

if not all_chunks and col.count() == 0:
    raise RuntimeError("nothing was loaded and the database is empty - nothing to answer from")

BATCH = 32
if todo:
    print(f"embedding {len(todo)} new chunks...")
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]

        res = client.embeddings.create(
            model=EMBED_MODEL,
            input=[c["text"] for c in batch],
        )
        vectors = [d.embedding for d in res.data]

        col.add(
            ids=[chunk_id(c) for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors,
            metadatas=[meta_of(c) for c in batch],
        )
        print(f"  {min(i+BATCH, len(todo))}/{len(todo)}")

print(f"collection ready - {col.count()} chunks from {len(SOURCES)} sources")


def generate_queries(question, n=4):
    out = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": (
                f"أعد صياغة السؤال التالي بـ {n} طرق مختلفة للبحث في قاعدة بيانات. "
                "استخدم مصطلحات ومرادفات متنوعة. "
                "اكتب كل صيغة في سطر منفصل، بدون ترقيم أو أي نص إضافي.\n\n"
                f"السؤال: {question}"
            ),
        }],
        temperature=0.7,
    )
    lines = [l.strip(" -•\t") for l in out.choices[0].message.content.splitlines()]
    return [question] + [l for l in lines if l][:n]


NAMED = re.compile("من|في|حسب|بحسب|وفق")
CUE = re.compile("اصدار|تقرير|ملف|مجل|فيديو|مقطع|وثيق")

def detect_source(question, threshold=0.75, cued=0.5):
    """If the question names one of the sources, return its id, else None.

    A question counts as naming a source only when most of that source's title
    appears in it, so topic words shared with a title - "الذكاء الاصطناعي" say -
    do not quietly narrow a broad question down to one document. Phrasing it as
    an explicit request ("من إصدار ..." / "ما ورد في تقرير ...") lowers the bar,
    because then the intent to pin one source is stated rather than guessed.
    """
    q = norm_ar(question)
    bar = cued if (NAMED.search(q) and CUE.search(q)) else threshold

    best, best_score = None, 0.0
    for s in SOURCES:
        words = {w for w in re.findall(r"\w+", norm_ar(s["title"])) if len(w) > 3}
        if not words:
            continue
        score = sum(1 for w in words if w in q) / len(words)
        if score > best_score:
            best, best_score = s["source_id"], score
    return best if best_score >= bar else None


def embed(texts):
    return [d.embedding for d in
            client.embeddings.create(model=EMBED_MODEL, input=texts).data]


def _merge(hits):
    best = {}
    for ids, docs, metas, dists in zip(
        hits["ids"], hits["documents"], hits["metadatas"], hits["distances"]
    ):
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            if cid not in best or dist < best[cid]["dist"]:
                best[cid] = {"doc": doc, "meta": meta, "dist": dist}
    return best


def multiquery_search(question, k=4, source_id=None):
    """Returns (rows, scoped_source_id). source_id pins the search to one source;
    without it the question itself is checked for a source name, and failing
    that the whole collection is searched as before."""
    sid = source_id or detect_source(question)

    if sid:
        queries = generate_queries(question)
        hits = col.query(query_embeddings=embed(queries),
                         n_results=k * 2, where={"source_id": sid})
        rows = sorted(_merge(hits).values(), key=lambda x: x["dist"])[:k * 3]
        print(f"[scoped] {sid} - {len(rows)} chunks ranked within the source")
        return rows, sid

    queries = generate_queries(question)
    print("queries:", *queries, sep="\n  ")
    hits = col.query(query_embeddings=embed(queries), n_results=k)
    return sorted(_merge(hits).values(), key=lambda x: x["dist"])[:k * 2], None


def ask(question, source_id=None):
    results, scope = multiquery_search(question, source_id=source_id)

    context = "\n\n═══\n\n".join(
        f"العنوان: {r['meta']['title']}\n"
        f"الموضع: {r['meta']['locator']}\n"
        + (f"الرابط: {r['meta']['url']}\n" if r["meta"].get("url") else "")
        + f"النص: {r['doc']}"
        for r in results
    )

    out = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content":
                "أجب على السؤال باستخدام السياق المُعطى فقط. "
                "بعد كل معلومة، اذكر المصدر بهذا الشكل: [العنوان - الموضع](الرابط) "
                "حيث العنوان والموضع والرابط منسوخة حرفياً من حقول السياق. "
                "الموضع يكون وقتاً في الفيديو أو رقم صفحة في الإصدار. "
                "إذا لم يكن للمقطع حقل رابط، اكتب الاستشهاد بدون رابط هكذا: "
                "[العنوان - الموضع] ولا تخترع رابطاً. "
                "لا تخترع رقماً غير موجود في النص. "
                "بعض النصوص مستخرجة من ملفات PDF وقد تحتوي على حروف مبتورة؛ "
                "اعتمد على المعنى ولا تنقل الأخطاء الإملائية في إجابتك. "
                "إذا لم يحتوِ السياق على الإجابة، قل ذلك."},
            {"role": "user", "content": f"السياق:\n{context}\n\nالسؤال: {question}"},
        ],
        temperature=0,
    )
    answer = out.choices[0].message.content
    print("\n" + answer)
    return answer, results, scope


if __name__ == "__main__":
    while True:
        q = input("\nسؤالك (أو اكتب خروج): ").strip()
        if q in {"خروج", "exit", "quit", ""}:
            break
        ask(q)
