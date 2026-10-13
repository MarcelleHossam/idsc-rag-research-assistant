"""Recover real Arabic text from IDSC PDFs whose ToUnicode cmap is corrupt.

The PDFs embed subsetted Tajawal faces with a broken ToUnicode map, so normal
extraction yields mojibake. The glyph outlines survive intact, so each glyph is
identified by hashing its outline against the original Google Fonts Tajawal
faces, which do carry a usable cmap.
"""
import io, os, glob, unicodedata, re
import pymupdf
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import DecomposingRecordingPen

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_REF = None
_TATWEEL = "\u0640"


def _outline_sig(glyphset, name):
    pen = DecomposingRecordingPen(glyphset)
    try:
        glyphset[name].draw(pen)
    except Exception:
        return None
    return repr(pen.value) if pen.value else None


def _reference():
    """outline signature -> unicode, built from the original Tajawal faces"""
    global _REF
    if _REF is not None:
        return _REF
    ref = {}
    for path in sorted(glob.glob(os.path.join(FONT_DIR, "Tajawal-*.ttf"))):
        ttf = TTFont(path)
        gs = ttf.getGlyphSet()
        g2u = {}
        for cp, name in ttf.getBestCmap().items():
            g2u.setdefault(name, chr(cp))
        for name in ttf.getGlyphOrder():
            u = g2u.get(name)
            if not u:
                continue
            sig = _outline_sig(gs, name)
            if sig:
                ref.setdefault(sig, u)
    _REF = ref
    return ref


def glyph_table(doc, page):
    """(face, gid) -> unicode for the Tajawal subsets used on this page"""
    ref = _reference()
    table = {}
    for f in page.get_fonts():
        xref, name = f[0], f[3]
        if "Tajawal" not in name:
            continue
        try:
            _, _, _, buf = doc.extract_font(xref)
            sub = TTFont(io.BytesIO(buf))
        except Exception:
            continue
        gs = sub.getGlyphSet()
        face = name.split("+")[-1]
        for gid, gname in enumerate(sub.getGlyphOrder()):
            sig = _outline_sig(gs, gname)
            u = ref.get(sig) if sig else None
            if u:
                table[(face, gid)] = u
    return table


_DOUBLE_ALEF = re.compile("([\u0627\u0623\u0625\u0622])\\1+")
_HAMZA_LIG = re.compile("([\u0623\u0625\u0622])(\u0644)(\\1)")
_LONE_LAM_ALEF = re.compile("(?<!\\w)\u0627\u0644\u0627(?!\\w)")


def _clean(s):
    s = unicodedata.normalize("NFKC", s).replace(_TATWEEL, "")
    s = _DOUBLE_ALEF.sub("\\1", s)
    s = _HAMZA_LIG.sub("\\2\\3", s)
    s = _LONE_LAM_ALEF.sub("\u0644\u0627", s)
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    return s


def page_text(doc, page):
    """Page text in logical order with glyphs recovered from the outlines.

    rawdict supplies reading order and line breaks; texttrace supplies the
    glyph ids. The two are joined on each character's origin.
    """
    table = glyph_table(doc, page)
    by_origin = {}
    for span in page.get_texttrace():
        face = str(span.get("font", "")).split("+")[-1]
        for ch in span["chars"]:
            uni, gid, origin = ch[0], ch[1], ch[2]
            fixed = table.get((face, gid))
            if fixed is not None:
                by_origin[(round(origin[0], 1), round(origin[1], 1), uni)] = fixed

    at_origin = {}
    for (ox, oy, _uni), fixed in by_origin.items():
        at_origin.setdefault((ox, oy), fixed)

    out = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            buf, prev = [], None
            for span in line["spans"]:
                for ch in span["chars"]:
                    ox, oy = round(ch["origin"][0], 1), round(ch["origin"][1], 1)
                    fixed = at_origin.get((ox, oy))
                    if fixed is None:
                        buf.append(ch["c"])
                        prev = None
                        continue
                    if (ox, oy) != prev:
                        buf.append(fixed)
                        prev = (ox, oy)
            txt = _clean("".join(buf)).strip()
            if txt:
                out.append(txt)
    return "\n".join(out)


def doc_pages(path):
    doc = pymupdf.open(path)
    for i in range(len(doc)):
        yield i + 1, page_text(doc, doc[i])
    doc.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else lo
    doc = pymupdf.open(path)
    for i in range(lo - 1, hi):
        print(f"===== page {i+1} =====")
        print(page_text(doc, doc[i]))
