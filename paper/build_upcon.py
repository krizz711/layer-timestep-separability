"""Render paper_upcon.md into an IEEE conference-format PDF for UPCON 2026.

Differences from ../build_pdf.py, all of them compliance requirements:

* IEEEtran conference page geometry on US Letter.  Margins 0.625 in left and
  right, 0.75 in top, 1.0 in bottom; two 3.5 in columns with a 0.25 in gutter.
* Times New Roman is EMBEDDED as a TrueType subset.  The previous build used
  reportlab's base-14 Times, which is referenced by name and not embedded, and
  IEEE PDF eXpress rejects non-embedded fonts at camera-ready.
* Body text at 10 pt, the template size, rather than 9.3 pt.
* A single-column abstract in bold at the head of column one, as IEEEtran sets
  it, instead of the three-column abstract band of the journal-style build.
* Section headings as centred Roman numerals in caps.
* No page numbers, per the IEEEtran conference default.

Usage:  python build_upcon.py [src.md] [out.pdf]
"""
import re
import os
import json
import struct
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
                                Spacer, Image, Table, TableStyle, KeepTogether,
                                FrameBreak, NextPageTemplate)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "fig")     # figures live one level up
# equations are re-set for this build with the disentangled notation; fall back
# to the shared set if that has not been rendered yet
EQDIR = os.path.join(HERE, "eqfig")
if not os.path.isdir(EQDIR):
    EQDIR = FIGDIR
SRC = os.path.join(HERE, "paper_upcon.md")
OUT = os.path.join(HERE, "layer_timestep_upcon.pdf")

# ---------------------------------------------------------------- fonts ----
# Embedded TrueType, so every glyph in the file is carried by the file.
WINF = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
SERIF, BOLD, ITAL, BITAL, MONO = "TNR", "TNR-Bd", "TNR-It", "TNR-BdIt", "CourierNew"
pdfmetrics.registerFont(TTFont(SERIF, os.path.join(WINF, "times.ttf")))
pdfmetrics.registerFont(TTFont(BOLD, os.path.join(WINF, "timesbd.ttf")))
pdfmetrics.registerFont(TTFont(ITAL, os.path.join(WINF, "timesi.ttf")))
pdfmetrics.registerFont(TTFont(BITAL, os.path.join(WINF, "timesbi.ttf")))
pdfmetrics.registerFont(TTFont(MONO, os.path.join(WINF, "cour.ttf")))
pdfmetrics.registerFontFamily(SERIF, normal=SERIF, bold=BOLD,
                              italic=ITAL, boldItalic=BITAL)

# ------------------------------------------------------------- geometry ----
PW, PH = letter                       # 612 x 792 pt
ML = MR = 0.625 * inch
MT = 0.75 * inch
MB = 1.00 * inch
COLW = 3.5 * inch
GUTTER = 0.25 * inch
TEXTW = 2 * COLW + GUTTER             # 7.25 in
assert abs(ML + TEXTW + MR - PW) < 0.5, "column geometry does not fill the page"

BODY, LEAD = 10.0, 11.6
# figure width as a fraction of the column; FIGSCALE=0.9 buys ~80 pt over five figures
FIGSCALE = float(os.environ.get("FIGSCALE", "0.97"))

S = {
    "title": ParagraphStyle("title", fontName=SERIF, fontSize=20, leading=23,
                            alignment=TA_CENTER, spaceAfter=10),
    "author": ParagraphStyle("author", fontName=SERIF, fontSize=11, leading=13.5,
                             alignment=TA_CENTER, spaceAfter=1),
    "affil": ParagraphStyle("affil", fontName=SERIF, fontSize=9.5, leading=11.5,
                            alignment=TA_CENTER, spaceAfter=1),
    "abstract": ParagraphStyle("abstract", fontName=BOLD, fontSize=9, leading=10.8,
                               alignment=TA_JUSTIFY, firstLineIndent=10,
                               spaceAfter=6),
    "kw": ParagraphStyle("kw", fontName=BOLD, fontSize=9, leading=10.8,
                         alignment=TA_JUSTIFY, firstLineIndent=10, spaceAfter=9),
    # keepWithNext stops a section head being left alone at the foot of a column
    "h1": ParagraphStyle("h1", fontName=SERIF, fontSize=10, leading=12,
                         alignment=TA_CENTER, spaceBefore=8, spaceAfter=4,
                         keepWithNext=1),
    "body": ParagraphStyle("body", fontName=SERIF, fontSize=BODY, leading=LEAD,
                           alignment=TA_JUSTIFY, firstLineIndent=10, spaceAfter=0),
    "body0": ParagraphStyle("body0", fontName=SERIF, fontSize=BODY, leading=LEAD,
                            alignment=TA_JUSTIFY, firstLineIndent=0, spaceAfter=0),
    "eqno": ParagraphStyle("eqno", fontName=SERIF, fontSize=9.5, leading=11,
                           alignment=TA_RIGHT),
    "cap": ParagraphStyle("cap", fontName=SERIF, fontSize=8, leading=9.5,
                          alignment=TA_JUSTIFY, spaceBefore=4, spaceAfter=6),
    "tnum": ParagraphStyle("tnum", fontName=SERIF, fontSize=8, leading=9.6,
                           alignment=TA_CENTER, spaceBefore=6, spaceAfter=1),
    "tcap": ParagraphStyle("tcap", fontName=SERIF, fontSize=8, leading=9.6,
                           alignment=TA_CENTER, spaceAfter=3),
    "ref": ParagraphStyle("ref", fontName=SERIF, fontSize=8, leading=9.2,
                          alignment=TA_JUSTIFY, leftIndent=11, firstLineIndent=-11,
                          spaceAfter=0.8),
    "item": ParagraphStyle("item", fontName=SERIF, fontSize=BODY, leading=LEAD,
                           alignment=TA_JUSTIFY, leftIndent=12, firstLineIndent=-12,
                           spaceAfter=1.5),
}

ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI",
         "XII", "XIII", "XIV", "XV"]


def png_aspect(path):
    d = open(path, "rb").read(33)
    w, h = struct.unpack(">II", d[16:24])
    return h / w


def md_inline(t):
    t = t.replace("&nbsp;", "")
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = t.replace("\u00a0", "&nbsp;")
    t = re.sub(r"(?<![A-Za-z0-9*])\*\*(?!\s)(.+?)(?<!\s)\*\*(?![A-Za-z0-9*])",
               r"<b>\1</b>", t)
    t = re.sub(r"(?<![A-Za-z0-9*])\*(?!\s)(.+?)(?<!\s)\*(?![A-Za-z0-9*])",
               r"<i>\1</i>", t)
    t = re.sub(r"`([^`]+)`", r"<font name='%s' size='8.6'>\1</font>" % MONO, t)
    sup = {"\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
           "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
           "\u2078": "8", "\u2079": "9", "\u207b": "-"}
    out, buf = [], []
    for ch in t:
        if ch in sup:
            buf.append(sup[ch])
        else:
            if buf:
                out.append("<super>" + "".join(buf) + "</super>")
                buf = []
            out.append(ch)
    if buf:
        out.append("<super>" + "".join(buf) + "</super>")
    return "".join(out)


def parse(path):
    raw = open(path, encoding="utf-8").read().split("\n")
    title, blocks, cur, key = None, [], [], None
    for ln in raw:
        if ln.startswith("# "):
            title = ln[2:].strip()
            continue
        m = re.match(r"^%%\s*(.+?)\s*$", ln)
        if m:
            if key:
                blocks.append((key, cur))
            key, cur = m.group(1), []
            continue
        if key:
            cur.append(ln)
    if key:
        blocks.append((key, cur))
    return title, blocks


def paras(lines):
    out, buf = [], []
    for ln in lines:
        if ln.strip():
            buf.append(ln.strip())
        elif buf:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return out


def render_body(lines):
    out, buf, mode, first = [], [], "body", True

    def flush():
        nonlocal buf, mode, first
        if not buf:
            return
        txt = md_inline(" ".join(buf))
        if mode == "item":
            out.append(Paragraph(txt, S["item"]))
        else:
            out.append(Paragraph(txt, S["body0"] if first else S["body"]))
            first = False
        buf, mode = [], "body"

    for ln in lines:
        if not ln.strip():
            # Only the paragraph that opens a block is set flush left: that is
            # either the first paragraph of a section or the sentence resuming
            # after a display equation.  Every later paragraph is indented, as
            # the IEEE template sets them.
            flush()
            continue
        if re.match(r"^\d+\.\s+\S", ln):
            flush()
            mode = "item"
            buf = [ln.strip()]
            continue
        buf.append(ln.strip())
    flush()
    return out


def make_table(lines, colw):
    body = [l for l in lines if l.strip()]
    cap, rows = [], []
    for l in body:
        (rows if "|" in l else cap).append(l)
    caption = " ".join(cap)
    grid = [[c.strip() for c in r.split("|")] for r in rows]
    ncol = max(len(r) for r in grid)
    grid = [r + [""] * (ncol - len(r)) for r in grid]
    fs = 7.6 if ncol <= 4 else (6.8 if ncol <= 5 else 6.2)
    cs = ParagraphStyle("c", fontName=SERIF, fontSize=fs, leading=fs + 1.6,
                        alignment=TA_CENTER)
    hs = ParagraphStyle("h", fontName=BOLD, fontSize=fs, leading=fs + 1.6,
                        alignment=TA_CENTER)
    data = [[Paragraph(md_inline(c), hs if i == 0 else cs) for c in r]
            for i, r in enumerate(grid)]
    from reportlab.pdfbase.pdfmetrics import stringWidth
    nat = []
    for j in range(ncol):
        w = max(stringWidth(re.sub(r"<[^>]+>", "", grid[i][j]),
                            BOLD if i == 0 else SERIF, fs) for i in range(len(grid)))
        nat.append(w + 7)
    scale = colw / sum(nat)
    ws = [n * scale for n in nat]
    t = Table(data, colWidths=ws, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.9, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.9, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return caption, t


def equation(n, meta):
    path = os.path.join(EQDIR, "eq%d.png" % n)
    w, h = meta["w"], meta["h"]
    numw = 24.0
    avail = COLW - numw - 4
    sc = min(1.0, avail / w)
    img = Image(path, width=w * sc, height=h * sc)
    img.hAlign = "CENTER"
    t = Table([[img, Paragraph("(%d)" % n, S["eqno"])]],
              colWidths=[avail, numw], rowHeights=[h * sc + 8])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    t.hAlign = "LEFT"
    return t


def heading(text):
    """IEEE section head: centred, Roman numeral, caps."""
    return Paragraph(md_inline(text.upper()), S["h1"])


def build():
    title, blocks = parse(SRC)
    bd = dict(blocks)
    eqs = json.load(open(os.path.join(EQDIR, "eqs.json"), encoding="utf-8"))

    # ---------- full-width title block ----------
    head = [Paragraph(md_inline(title), S["title"])]
    alines = [l.strip() for l in bd["AUTHOR"] if l.strip()]
    for i, ln in enumerate(alines):
        head.append(Paragraph(md_inline(ln), S["author"] if i == 0 else S["affil"]))
    th = sum(f.wrap(TEXTW, PH)[1] + f.getSpaceAfter() for f in head) + 12

    def F(x, y, w, h, i):
        return Frame(x, y, w, h, id=i, leftPadding=0, rightPadding=0,
                     topPadding=0, bottomPadding=0)

    top = F(ML, PH - MT - th, TEXTW, th, "top")
    ch1 = PH - MT - th - MB                       # column height on page 1
    chr_ = PH - MT - MB                           # column height on later pages
    c1 = F(ML, MB, COLW, ch1, "c1")
    c2 = F(ML + COLW + GUTTER, MB, COLW, ch1, "c2")
    r1 = F(ML, MB, COLW, chr_, "r1")
    r2 = F(ML + COLW + GUTTER, MB, COLW, chr_, "r2")

    doc = BaseDocTemplate(OUT, pagesize=letter, title=title,
                          author="Munal Singh, Marina Thottakkara, Kashikaa, Deepa D",
                          subject="UPCON 2026 submission",
                          leftMargin=ML, rightMargin=MR,
                          topMargin=MT, bottomMargin=MB)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[top, c1, c2]),
        PageTemplate(id="rest", frames=[r1, r2]),
    ])

    story = [NextPageTemplate("rest")] + list(head) + [FrameBreak()]

    story.append(Paragraph(
        "<i><b>Abstract</b></i>&mdash;" + md_inline(" ".join(paras(bd["ABSTRACT"]))),
        S["abstract"]))
    story.append(Paragraph(
        "<i><b>Index Terms</b></i>&mdash;" + md_inline(" ".join(paras(bd["KEYWORDS"]))),
        S["kw"]))

    tbl_no = {"TABLE1": "I", "TABLE2": "II", "TABLE3": "III",
              "TABLE4": "IV", "TABLE5": "V"}
    for key, lines in blocks:
        if key in ("AUTHOR", "ABSTRACT", "KEYWORDS"):
            continue
        if key.startswith("TABLE"):
            cap, t = make_table(lines, COLW)
            story.append(KeepTogether([
                Paragraph("TABLE %s" % tbl_no.get(key, "?"), S["tnum"]),
                Paragraph(md_inline(cap), S["tcap"]), t, Spacer(1, 6)]))
            continue
        if key == "ACKNOWLEDGMENT":
            story.append(heading("Acknowledgment"))
            story.extend(render_body(lines))
            continue
        if key == "REFERENCES":
            story.append(heading("References"))
            for ln in lines:
                if ln.strip():
                    story.append(Paragraph(md_inline(ln.strip()), S["ref"]))
            continue
        m = re.match(r"^EQ\s*(\d+)$", key)
        if m:
            n = int(m.group(1))
            story.append(Spacer(1, 2))
            story.append(equation(n, eqs[str(n)]))
            story.append(Spacer(1, 3))
            continue
        if key.startswith("FIG:"):
            rel = key.split(":", 1)[1].strip()
            # figures whose artwork carries notation are re-rendered for this
            # build into figdir/; everything else comes from the shared set
            local = os.path.join(HERE, "figdir", os.path.basename(rel))
            path = local if os.path.exists(local) else                 os.path.join(os.path.dirname(HERE), rel)
            if not os.path.exists(path):
                raise SystemExit("missing figure %s" % path)
            cap = " ".join(l.strip() for l in lines if l.strip())
            story.append(KeepTogether([
                Spacer(1, 4),
                Image(path, width=FIGSCALE * COLW,
                      height=FIGSCALE * COLW * png_aspect(path)),
                Paragraph(md_inline(cap), S["cap"])]))
            continue
        if key == "CONT":
            story.extend(render_body(lines))
            continue
        m = re.match(r"^S(\d+)\.\s*(.+)$", key)
        if m:
            story.append(heading("%s.  %s" % (ROMAN[int(m.group(1))], m.group(2))))
        story.extend(render_body(lines))

    doc.build(story)

    # reportlab declares a base-14 Helvetica in every page's resources even when
    # no glyph is drawn with it.  It is not embedded, and IEEE PDF eXpress reads
    # the resource list rather than the drawn glyphs, so the unused entry is
    # collected away here.
    import fitz
    d = fitz.open(OUT)
    tmp = OUT + ".tmp"
    d.save(tmp, garbage=4, clean=True, deflate=True)
    d.close()
    os.replace(tmp, OUT)

    d = fitz.open(OUT)
    fonts = {(f[3], f[1]) for i in range(d.page_count) for f in d.get_page_fonts(i)}
    bad = sorted(n for n, ext in fonts if ext == "n/a")
    print("wrote %s  pages=%d  size=%.2f MB"
          % (OUT, d.page_count, os.path.getsize(OUT) / 1e6))
    print("fonts: %d embedded, non-embedded: %s"
          % (len(fonts) - len(bad), bad if bad else "none"))
    if d.page_count > 6:
        print("WARNING: over the UPCON 6-page limit")
    return d.page_count


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        SRC, OUT = sys.argv[1], sys.argv[2]
    build()
