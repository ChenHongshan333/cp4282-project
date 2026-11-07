"""Static Figures 2 and 9 of report.md, rendered with headless Chrome.

Figure 2 is the roadmap SVG of docs/index.html (read only), with the CSS rules and the
highlight step of the website applied. Figure 9 is drawn here with the same structure and
a short finding in every box (the findings are the rm-* cards of docs/index.html, shortened).

Run from the repository root:  python report_assets/make_roadmaps.py
Writes report_assets/figure2_roadmap.{html,png} and report_assets/figure9_findings.{html,png}.
"""
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_assets"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
PROFILE = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / ".chrome-profile"
SCALE = 3

# Colours and fonts of docs/assets/style.css (:root), resolved so the pages stand alone.
VARS = {"--ink": "#141414", "--ink-2": "#3d3c39", "--muted": "#6b6a65", "--accent": "#1c5cab"}
FONTS = ('<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@'
         '0,8..60,400;0,8..60,600;0,8..60,650;0,8..60,700;1,8..60,400&display=swap" rel="stylesheet">')
# The roadmap rules of docs/assets/style.css (figure.roadmap ...), with the variables filled in.
CSS = """
html, body { margin: 0; background: #ffffff; }
svg { display: block; }
text { font-family: "Source Serif 4", "Source Serif Pro", Georgia, serif; }
.rm-name { font-size: 14px; font-weight: 650; fill: var(--ink); }
.rm-q { font-size: 12.5px; fill: var(--ink-2); }
.rm-lane { font-size: 12px; font-weight: 600; letter-spacing: 0.04em; fill: var(--muted); }
.rm-side { fill: var(--muted); }
.rm-name.rm-side { fill: var(--ink-2); font-weight: 600; }
.rm-lane-main { fill: var(--accent); }
.rm-hl { fill: var(--ink); }
.rm-qtag { font-weight: 700; }
.rm-hl-bg { fill: #f1eff8; }
/* Figure 9 only */
.rm-find { font-size: 12.5px; fill: var(--ink-2); }
.rm-ev { font-size: 11.5px; font-style: italic; fill: var(--muted); }
"""
CSS = ":root {" + " ".join(f"{k}: {v};" for k, v in VARS.items()) + "}\n" + CSS

# The highlight step of docs/assets/js/main.js (highlightRoadmap), and a check that every
# text fits inside its box; the result is written into the page for --dump-dom.
SCRIPT = """
document.fonts.ready.then(() => {
  for (const t of document.querySelectorAll("text.rm-hl")) {
    const b = t.getBBox();
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    Object.entries({ x: b.x - 5, y: b.y - 2, width: b.width + 10, height: b.height + 4, rx: 3, class: "rm-hl-bg" })
      .forEach(([k, v]) => r.setAttribute(k, v));
    t.before(r);
  }
  const boxes = [...document.querySelectorAll("rect[data-box]")].map((r) => r.getBBox());
  const bad = [];
  for (const t of document.querySelectorAll("text[data-in]")) {
    const b = t.getBBox(), box = boxes[+t.dataset.in];
    if (b.x < box.x + 4 || b.x + b.width > box.x + box.width - 4 || b.y < box.y || b.y + b.height > box.y + box.height)
      bad.push(t.textContent);
  }
  const fam = getComputedStyle(document.querySelector("text")).fontFamily;
  const ok = document.fonts.check('14px "Source Serif 4"');
  document.body.setAttribute("data-check", JSON.stringify({ fontLoaded: ok, family: fam, overflow: bad }));
});
"""


def page(svg, width, height):
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>roadmap</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
{FONTS}
<style>{CSS}</style></head>
<body><div style="width:{width}px;height:{height}px">{svg}</div>
<script>{SCRIPT}</script></body></html>
"""


def mark_boxes(svg):
    """Tag every box rect and the texts drawn inside it, for the overflow check."""
    boxes = []
    def rect(m):
        a = dict(re.findall(r'(\w[\w-]*)="([^"]*)"', m.group(0)))
        if float(a.get("height", 0)) < 30 or a.get("fill") in ("#e8f0fb", "#f3f2ee"):
            return m.group(0)                      # legend swatches and lane backgrounds
        boxes.append(tuple(float(a[k]) for k in ("x", "y", "width", "height")))
        return m.group(0).replace("<rect ", f'<rect data-box="{len(boxes) - 1}" ', 1)
    svg = re.sub(r"<rect [^>]*?/>", rect, svg)
    def text(m):
        x, y = float(m.group(1)), float(m.group(2))
        for i, (bx, by, bw, bh) in enumerate(boxes):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return m.group(0).replace("<text ", f'<text data-in="{i}" ', 1)
        return m.group(0)
    return re.sub(r'<text x="([\d.]+)" y="([\d.]+)"', text, svg)


# ------------------------------------------------------------------------------- Figure 2
def figure2():
    src = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    i = src.index('<figure class="roadmap"')
    svg = src[src.index("<svg", i): src.index("</svg>", i) + len("</svg>")]
    svg = svg.replace('style="max-width:734px;margin:0 auto"', 'width="734" height="650"')
    # Degree 3 and 7 was planned in the proposal as a supplementary extension (report Section 3.1),
    # so its box gets a solid border, as the legend uses for planned steps.
    d = svg.index('class="rm-name rm-side">Degree 3 and 7')
    r = svg.rindex("<rect ", 0, d)
    box = svg[r: svg.index("/>", r) + 2]
    assert 'stroke-dasharray="5 4"' in box
    svg = svg[:r] + box.replace(' stroke-dasharray="5 4"', "") + svg[r + len(box):]
    return mark_boxes(svg), 734, 650


# ------------------------------------------------------------------------------- Figure 9
# name, Q tag, evidence (short), finding lines.  Source: the rm-* cards of docs/index.html.
MAIN = [
    ("Synthetic discs", None, "Preliminary diagnostic",
     ["All three abilities beat M0, but M1's gain fell from +1.27",
      "to −0.02 dB at a lower velocity learning rate, so that rate",
      "was tuned for every model."]),
    ("Tuning", None, "Validation clip",
     ["Every model, the baseline included, uses the best setting",
      "found in the tested range; best and worst differ by 0.3 to 0.6 dB."]),
    ("Main experiment, 3,000 steps", "Q1", "Reliable",
     ["M2 and M3 beat M0 by 0.5 to 2.3 dB and pass the test;",
      "M1 fails, and its gain lies mostly in the background."]),
    ("More Gaussians; longer training", None, "Fairly reliable",
     ["No ability keeps half of its lead at 10,000 steps: the early",
      "gains count as faster learning."]),
    ("Re-tuning for full training", None, "Validation clip",
     ["Every model settles after 18,000 to 24,000 steps; the",
      "learning rate moves the result by only about 0.1 dB."]),
    ("Main experiment, fully trained", "Q1, Q2", "Reliable",
     ["The baseline catches up: most cells are within ±0.3 dB,",
      "and all three abilities fail the test."]),
]
SIDE = {  # main-line row it branches from
    2: ("Still photographs", "Q2", "Reliable",
        ["Shrinking gains 0.6 to 1.9 dB on still photographs,",
         "about as much as on video: the early gains appear",
         "without motion."]),
    5: ("Degree 3 and 7", None, "Fairly reliable",
        ["D3 is worse than M1 in 7 of 8 cells; D7 is worse",
         "everywhere, by 1.1 to 2.3 dB on the fitted frames."]),
}
Q3 = [
    ("Frames between", "Exploratory",
     ["M1 and M3 draw the dog 0.2 to",
      "0.5 dB better between frames,",
      "level on the fitted frames."]),
    ("Synthetic arcs", "Reliable",
     ["The gain between frames grows",
      "with the bend, up to about 9 dB;",
      "the Gaussians bend with the motion."]),
    ("Confirmatory test", "Fairly reliable",
     ["The moving object is about 0.5 dB",
      "better between frames; 9 of 12 cells",
      "pass. No consistent whole-frame gain."]),
]
PLANNED = {"Synthetic discs", "Tuning", "Main experiment, 3,000 steps", "Degree 3 and 7"}


def esc(s):
    return html.escape(s, quote=False)


def figure9():
    W = 940
    ARROW = ('<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
             '<path d="M0 0L10 5L0 10z" fill="#8f8d86"/></marker>'
             '<marker id="am" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
             '<path d="M0 0L10 5L0 10z" fill="#141414"/></marker></defs>')
    out = []
    mx, mw, sx, sw = 28, 430, 498, 414          # main and side box x and width
    top, gap, pad = 44, 22, 12

    def hbox(n):                                 # box height for n finding lines
        return 24 + 6 + 16 * n + 10

    def box(x, y, w, h, main, planned):
        if main:
            dash = "" if planned else ' stroke-dasharray="6 4"'
            return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#ffffff" stroke="#141414" stroke-width="1.6"{dash}/>'
        dash = "" if planned else ' stroke-dasharray="5 4"'
        return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="#fbfaf8" stroke="#8f8d86" stroke-width="1.2"{dash}/>'

    def content(x, y, w, name, tag, ev, lines, side):
        s = " rm-side" if side else ""
        t = f'<text x="{x + pad}" y="{y + 21}" class="rm-name{s}">{esc(name)}'
        if tag:
            t += f'<tspan class="rm-qtag" dx="8" style="fill:var(--accent)">{tag}</tspan>'
        t += "</text>"
        t += f'<text x="{x + w - pad}" y="{y + 21}" class="rm-ev" text-anchor="end">{esc(ev)}</text>'
        for k, line in enumerate(lines):
            t += f'<text x="{x + pad}" y="{y + 42 + 16 * k}" class="rm-find">{esc(line)}</text>'
        return t

    # main line
    ys, y = [], top
    for name, tag, ev, lines in MAIN:
        h = hbox(max(len(lines), len(SIDE.get(len(ys), (0, 0, 0, []))[3])))
        ys.append((y, h))
        y += h + gap
    main_bottom = ys[-1][0] + ys[-1][1]
    out.append(f'<rect x="14" y="10" width="{mx + mw + 14 - 14}" height="{main_bottom + 14 - 10}" rx="10" fill="#e8f0fb" stroke="#bcd2ef"/>')
    out.append(f'<text x="{mx}" y="31" class="rm-lane rm-lane-main">Main line</text>')
    out.append(f'<text x="{sx}" y="31" class="rm-lane">Diagnostics and extensions</text>')
    for k, ((name, tag, ev, lines), (y, h)) in enumerate(zip(MAIN, ys)):
        out.append(box(mx, y, mw, h, True, name in PLANNED))
        out.append(content(mx, y, mw, name, tag, ev, lines, False))
        if k:
            py, ph = ys[k - 1]
            out.append(f'<line x1="{mx + mw / 2}" y1="{py + ph}" x2="{mx + mw / 2}" y2="{y - 2}" stroke="#141414" stroke-width="2" marker-end="url(#am)"/>')
        if k in SIDE:
            sname, stag, sev, slines = SIDE[k]
            out.append(box(sx, y, sw, h, False, sname in PLANNED))
            out.append(content(sx, y, sw, sname, stag, sev, slines, True))
            out.append(f'<line x1="{mx + mw}" y1="{y + h / 2}" x2="{sx - 2}" y2="{y + h / 2}" stroke="#8f8d86" stroke-width="1.3" marker-end="url(#a)"/>')

    # Q3 lane
    ly = main_bottom + 26
    qh = hbox(max(len(q[2]) for q in Q3))
    qw, qgap = 278, 22
    lane_h = 34 + qh + 14
    out.append(f'<rect x="14" y="{ly}" width="{W - 28}" height="{lane_h}" rx="10" fill="#f3f2ee"/>')
    out.append(f'<text x="104" y="{ly + 20}" class="rm-lane rm-hl"><tspan class="rm-qtag">Q3</tspan>: the frames between</text>')
    out.append(f'<line x1="84" y1="{main_bottom}" x2="84" y2="{ly + 32}" stroke="#141414" stroke-width="2" marker-end="url(#am)"/>')
    for k, (name, ev, lines) in enumerate(Q3):
        x = 28 + k * (qw + qgap)
        out.append(box(x, ly + 34, qw, qh, False, False))
        out.append(content(x, ly + 34, qw, name, None, ev, lines, True))
        if k:
            out.append(f'<line x1="{x - qgap}" y1="{ly + 34 + qh / 2}" x2="{x - 2}" y2="{ly + 34 + qh / 2}" stroke="#8f8d86" stroke-width="1.3" marker-end="url(#a)"/>')

    # legend
    gy = ly + lane_h + 16
    out.append(f'<rect x="28" y="{gy}" width="26" height="14" rx="3" fill="#fff" stroke="#141414" stroke-width="1.4"/>'
               f'<text x="62" y="{gy + 11}" class="rm-q">planned in the proposal</text>'
               f'<rect x="228" y="{gy}" width="26" height="14" rx="3" fill="#fff" stroke="#141414" stroke-width="1.4" stroke-dasharray="5 4"/>'
               f'<text x="262" y="{gy + 11}" class="rm-q">added to answer a question raised by earlier results</text>'
               f'<text x="{W - 28}" y="{gy + 11}" class="rm-ev" text-anchor="end">italic: kind and strength of the evidence</text>')
    H = gy + 30
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
           f'aria-label="Roadmap of the experiments with the finding of each step">{ARROW}{"".join(out)}</svg>')
    return mark_boxes(svg), W, H


def render(name, svg, w, h):
    f = OUT / f"{name}.html"
    f.write_text(page(svg, w, h), encoding="utf-8")
    common = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
              f"--user-data-dir={PROFILE}", "--virtual-time-budget=10000", f"--window-size={w},{h}"]
    dom = subprocess.run(common + ["--dump-dom", f.as_uri()], capture_output=True, text=True, encoding="utf-8").stdout
    m = re.search(r'data-check="([^"]*)"', dom)
    print(name, html.unescape(m.group(1)) if m else "no check result")
    subprocess.run(common + [f"--force-device-scale-factor={SCALE}", f"--screenshot={OUT / (name + '.png')}", f.as_uri()],
                   capture_output=True)


if __name__ == "__main__":
    render("figure2_roadmap", *figure2())
    render("figure9_findings", *figure9())
