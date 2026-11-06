// The interactive parts of the project page: tables from data/tables.json and data/appendix_tables.json,
// one Gaussian in the volume of the video, the reading guides of the side columns, the term cards and the roadmap.
// (The training-curve, model-player and arc demos were removed in step 57; the renderer they used,
// gauss.js and gl.js, stays for volume.js and check.mjs.)

import { volumeWidget } from "./volume.js?v=8";

const $ = (s, root = document) => root.querySelector(s);
const $$ = (s, root = document) => [...root.querySelectorAll(s)];
const MINUS = "−";

// "no-cache" revalidates with the server, so a re-exported file is never read from a stale cache.
const json = (url) => fetch(url, { cache: "no-cache" }).then((r) => { if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); });
const DATA = { tables: json("data/tables.json"), appendix: json("data/appendix_tables.json") };

// ===================================================================================== tables
async function fillTables() {
  const tables = await DATA.tables;
  for (const el of $$("[data-table]")) {
    const t = tables[el.dataset.table];
    if (!t) continue;
    el.classList.add("table-wrap");
    // data-shade: grey out gains within ±0.3 dB, the margin of the pre-registered test
    const shade = "shade" in el.dataset;
    const cell = (c) => {
      const text = String(c).replace(/✓/g, '<span class="pass">✓</span>').replace(/✗/g, '<span class="fail">✗</span>')
        .replace(/^passes$/, '<span class="pass">passes</span>').replace(/^fails$/, '<span class="fail">fails</span>');
      const classes = [];
      if (/^[\d+−\-—]/.test(String(c))) classes.push("num");
      if (shade && Math.abs(parseFloat(String(c).replace(MINUS, "-"))) <= 0.3) classes.push("small");
      return `<td${classes.length ? ` class="${classes.join(" ")}"` : ""}>${text}</td>`;
    };
    el.innerHTML = `<table><caption>${el.dataset.caption || ""}</caption><thead><tr>${t.head
      .map((h, i) => `<th${i > 0 && /^[\d+−\-—]/.test(String(t.rows[0]?.[i] ?? "")) ? ' class="num"' : ""}>${h}</th>`)
      .join("")}</tr></thead><tbody>${t.rows.map((r) => `<tr>${r.map(cell).join("")}</tr>`).join("")}</tbody></table>`;
  }
  // Appendix tables (data-atable, stg/appendix_tables.py): one statistic per cell, header rows with spans,
  // group rows ({g}) for the clip or condition, and cells that meet the criterion of Section 3.3 ({ok}).
  const atables = await DATA.appendix;
  for (const el of $$("[data-atable]")) {
    const t = atables[el.dataset.atable];
    if (!t) continue;
    const n = t.num.length;
    const th = (h) => `<th${h.cs > 1 ? ` colspan="${h.cs}"` : ""}${h.rs > 1 ? ` rowspan="${h.rs}"` : ""}${
      h.cs > 1 ? ' class="span"' : h.num ? ' class="num"' : ""}>${h.t}</th>`;
    const td = (c, i, ok) => {
      const cls = [t.num[i] ? "num" : "", ok.includes(i) ? "meet" : ""].filter(Boolean).join(" ");
      return `<td${cls ? ` class="${cls}"` : ""}>${c}</td>`;
    };
    el.innerHTML = `<table class="data"><caption>${el.dataset.caption || ""}</caption><thead>${
      t.head.map((r) => `<tr>${r.map(th).join("")}</tr>`).join("")}</thead><tbody>${
      t.rows.map((r) => (r.g != null ? `<tr class="group"><td colspan="${n}">${r.g}</td></tr>`
        : `<tr>${r.c.map((c, i) => td(c, i, r.ok || [])).join("")}</tr>`)).join("")}</tbody></table>`;
  }
  // A long caption inside a table makes the table as wide as the caption. Put each table and its
  // caption in a box as wide as the table (.tbox, centred); the caption wraps at that width.
  for (const table of $$("table")) {
    const box = document.createElement("div");
    box.className = "tbox";
    table.before(box);
    const caption = $("caption", table);
    if (caption) {
      const div = document.createElement("div");
      div.className = "tcap";
      div.innerHTML = caption.innerHTML;
      box.appendChild(div);
      caption.remove();
    }
    box.appendChild(table);
  }
  gainScale();
  // the tables change the height of the page: go to the linked section again
  if (location.hash) $(location.hash)?.scrollIntoView();
}

// A soft background scale for gains over M0 (steps 60-61), only in the columns a table names in data-gain-cols
// (0-based cell indices of its data rows). Five fixed bins in dB, the same for every table; it encodes the size
// and direction of a gain, not the test of Section 3.3. Group rows and missing values ("—") stay plain.
// The legend is written once in the page (#colour-scale, before Appendix A); this function only colours cells.
const GAIN_BINS = (x) => (x <= -1 ? "gs-n2" : x < -0.3 ? "gs-n1" : x <= 0.3 ? "" : x < 1 ? "gs-p1" : "gs-p2");
const gainValue = (text) => {
  const t = text.trim().replace(/−/g, "-");
  return /^[+-]?\d+(\.\d+)?$/.test(t) ? parseFloat(t) : NaN;
};
function gainScale() {
  for (const wrap of $$("[data-gain-cols]")) {
    const cols = wrap.dataset.gainCols.split(",").map(Number);
    for (const row of $$("tbody tr:not(.group)", wrap)) {
      for (const i of cols) {
        const cell = row.cells[i];
        const x = cell ? gainValue(cell.textContent) : NaN;
        const bin = Number.isNaN(x) ? "" : GAIN_BINS(x);
        if (bin) cell.classList.add("gs", bin);
      }
    }
  }
}

// ===================================================================================== table of contents
// The left column (and the top bar on narrow screens) lists the sections. It is drawn as one space-time
// Gaussian, as in Section 1: the page is the time axis, each section a frame. The centre follows a curved
// path; a dot on it marks the reading position, and the slices along it fade and shrink away from the dot.
// The right column lists the headings of the section being read, then its demos, figures and tables.
// Both follow the scroll position.
const BONE_X0 = 36, BONE_SWAY = 13;
const boneX = (u) => BONE_X0 + BONE_SWAY * Math.sin(1.15 * u);

function drawBone(svg, nodes, uDot, hover) {
  const { width: W, height: H } = svg.getBoundingClientRect();
  if (!W || !H) return;
  const n = nodes.length;
  const boneY = (u) => {   // y of the path at parameter u (u = i at section i), continued past both ends
    const i = Math.max(0, Math.min(n - 2, Math.floor(u)));
    return nodes[i] + (u - i) * (nodes[i + 1] - nodes[i]);
  };
  const ell = (u, k, attrs) => {
    const x = boneX(u), y = boneY(u);
    return `<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="${(13 * k).toFixed(1)}" ry="${(7 * k).toFixed(1)}" transform="rotate(-30 ${x.toFixed(1)} ${y.toFixed(1)})" ${attrs}/>`;
  };
  const w = (u) => Math.exp(-0.5 * ((u - uDot) / 1.1) ** 2);   // the weight of the slice, peaked at the reading position
  let s = "";
  // the path of the centre
  const pts = [];
  for (let u = -0.45; u <= n - 1 + 0.45; u += 0.05) pts.push(`${boneX(u).toFixed(1)} ${boneY(u).toFixed(1)}`);
  s += `<path d="M${pts.join("L")}" fill="none" stroke="#1b1b1b" stroke-width="1.1" stroke-dasharray="4 3" opacity="0.7"/>`;
  // the Gaussian between the frames: one outline every eighth of a section, shrinking as it fades
  for (let u = -0.375; u <= n - 1 + 0.375; u += 0.125) {
    const k = w(u);
    s += ell(u, Math.max(k, 0.4), `fill="#2a78d6" fill-opacity="${(0.04 * k).toFixed(3)}" stroke="#2a78d6" stroke-width="0.9" stroke-opacity="${(0.15 + 0.6 * k).toFixed(2)}"`);
  }
  // the frames: one plane per section; the section being read is the orange one
  const cur = Math.max(0, Math.min(n - 1, Math.floor(uDot + 1e-6)));
  nodes.forEach((y, i) => {
    const x = boneX(i), on = i === cur, hot = i === hover;
    const plane = [[x - 15, y - 21], [x + 15, y - 13], [x + 15, y + 21], [x - 15, y + 13]].map((p) => p.map((v) => v.toFixed(1)).join(" "));
    s += `<path d="M${plane.join("L")}Z" fill="${on ? "#d4521d" : "#8f8d86"}" fill-opacity="${on ? 0.1 : hot ? 0.08 : 0.03}" stroke="${on ? "#d4521d" : "#8f8d86"}" stroke-width="${on || hot ? 1.4 : 0.8}" stroke-opacity="${on || hot ? 1 : 0.5}"/>`;
    const k = Math.max(w(i), 0.4);
    s += on ? ell(i, k, `fill="#d4521d" fill-opacity="0.75" stroke="#d4521d" stroke-width="1.4"`)
            : ell(i, k, `fill="${i < cur ? "#2a78d6" : "none"}" fill-opacity="${(0.15 + 0.5 * w(i)).toFixed(2)}" stroke="#2a78d6" stroke-width="1" stroke-opacity="${(0.3 + 0.6 * w(i)).toFixed(2)}"`);
  });
  // the reading position
  s += `<circle cx="${boneX(uDot).toFixed(1)}" cy="${boneY(uDot).toFixed(1)}" r="3.2" fill="#1b1b1b"/>`;
  svg.innerHTML = s;
}

function setupToc() {
  const links = $$("#toc a, #toc-top a");
  const sideLinks = $$("#toc a");
  const sections = $$("section.chapter[id]");
  const bone = $(".side-left .bone");
  const otp = $("#otp"), its = $("#its"), itsTitle = $(".side-title-2");
  const slug = (t) => t.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  sections.forEach((sec) => sec.querySelectorAll("h3, h4").forEach((h) => { if (!h.id) h.id = `${sec.id}-${slug(h.textContent)}`; }));
  let current = null, scope = null, heads = [], nodes = [], hover = -1, frame = 0;

  // place the section names down the whole column, a frame each
  const layout = () => {
    const H = bone.getBoundingClientRect().height, n = sideLinks.length;
    const gap = Math.min(110, (H - 60) / (n - 1)), top = (H - gap * (n - 1)) / 2;
    nodes = sideLinks.map((_, i) => top + i * gap);
    sideLinks.forEach((a, i) => { a.style.top = `${nodes[i]}px`; });
  };
  sideLinks.forEach((a, i) => {
    a.addEventListener("mouseenter", () => { hover = i; update(); });
    a.addEventListener("mouseleave", () => { hover = -1; update(); });
  });

  // the demos, figures and tables of a section, outside the folded details, each as a small card:
  // a thumbnail (the figure itself, or an icon), the kind and number, and a short name (data-short)
  const ICON_DEMO = `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="11" fill="#d4521d"/><path d="M9.5 7.5v9l7-4.5z" fill="#fff"/></svg>`;
  const ICON_TABLE = `<svg viewBox="0 0 24 24" fill="none" stroke="#6b6a65" stroke-width="1.4"><rect x="3" y="5" width="18" height="14" rx="1.5"/><path d="M3 9.5h18M3 14h18M10 5v14"/></svg>`;
  const contents = (sec) => [...sec.querySelectorAll(".widget, figure, .table-wrap")].filter((el) => !el.closest("details")).map((el) => {
    if (el.classList.contains("widget")) {
      const t = el.querySelector(".wtitle")?.textContent.replace(/^Try it:\s*/, "") ?? "";
      return { id: el.id, kind: "Interactive", name: t.charAt(0).toUpperCase() + t.slice(1), thumb: ICON_DEMO, demo: true };
    }
    const cap = el.querySelector(".tcap, figcaption, caption") ?? { textContent: el.dataset.caption ? el.dataset.caption.replace(/<[^>]+>/g, "") : "" };
    const m = cap.textContent.match(/^(Figure|Table) (A?\d+)\./);
    if (!m) return null;
    if (!el.id) el.id = `${m[1].toLowerCase()}-${m[2]}`;
    const img = el.querySelector("img");
    return { id: el.id, kind: `${m[1]} ${m[2]}`, name: el.dataset.short ?? "", thumb: img ? `<img src="${img.getAttribute("src")}" alt="">` : ICON_TABLE };
  }).filter((x) => x && x.id);

  const update = () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => {
      const line = innerHeight * 0.3;
      let idx = 0;
      sections.forEach((s, i) => { if (s.getBoundingClientRect().top <= line) idx = i; });
      const sec = sections[idx];
      if (sec !== current) {
        current = sec;
        links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === "#" + sec.id));
        // only the main headings; a heading may give a shorter name for this list (data-nav).
        // A box with data-nav (the answers of Section 6) is listed too, without a heading in the text.
        heads = [...sec.querySelectorAll("h3, .callout[data-nav]")];
        otp.innerHTML = heads.map((h) => `<a href="#${h.id}">${h.dataset.nav ?? h.textContent}</a>`).join("");
        otp.previousElementSibling.style.display = otp.style.display = heads.length ? "" : "none";
        otp.parentElement.scrollTop = 0;   // a new section starts at the top of the right column
      }
      // "In this section" lists the section's figures, tables and demos; in the appendix, only those of
      // the group (.app-group) being read
      const groups = [...sec.querySelectorAll(":scope > .app-group")];
      let sc = sec;
      if (groups.length) { sc = groups[0]; for (const g of groups) if (g.getBoundingClientRect().top <= line) sc = g; }
      if (sc !== scope) {
        scope = sc;
        const items = contents(sc);
        its.innerHTML = items.map((c) => `<a class="card" href="#${c.id}"><span class="thumb${c.demo ? " demo" : ""}">${c.thumb}</span><span class="txt"><span class="kind${c.demo ? " demo" : ""}">${c.kind}</span><span class="name">${c.name}</span></span></a>`).join("");
        itsTitle.style.display = its.style.display = items.length ? "" : "none";
        itsTitle.style.marginTop = heads.length ? "" : "5px";
      }
      let head = null;
      for (const h of heads) if (h.getBoundingClientRect().top <= line) head = h;
      otp.querySelectorAll("a").forEach((a) => a.classList.toggle("active", head !== null && a.getAttribute("href") === "#" + head.id));
      // reading position: the section index plus how far through it the reader is
      const r = sec.getBoundingClientRect();
      const uDot = idx + Math.max(0, Math.min(0.999, (line - r.top) / r.height));
      if (bone && nodes.length) drawBone(bone, nodes, uDot, hover);
    });
  };
  addEventListener("scroll", update, { passive: true });
  addEventListener("resize", () => { layout(); update(); });
  layout();
  update();
}

// ===================================================================================== term cards
// A term in the text (.term, data-term) opens a short note from #term-defs just above it, or below it
// when there is no room. A second click, a click elsewhere or Escape closes it.
function setupTerms() {
  const defs = $("#term-defs");
  if (!defs) return;
  const pop = document.createElement("div");
  pop.className = "term-pop";
  pop.setAttribute("role", "dialog");
  pop.hidden = true;
  document.body.appendChild(pop);
  let open = null;
  const close = () => { if (open) open.setAttribute("aria-expanded", "false"); open = null; pop.hidden = true; };
  const show = (term) => {
    const def = $(`[data-def="${term.dataset.term}"]`, defs);
    if (!def) return;
    pop.innerHTML = def.innerHTML;
    pop.hidden = false;
    const r = term.getBoundingClientRect(), w = pop.offsetWidth, h = pop.offsetHeight;
    // stay inside the column of the text, clear of the side columns
    const col = term.closest("section.chapter")?.getBoundingClientRect() ?? { left: 12, right: innerWidth - 12 };
    const left = Math.max(col.left, Math.min(col.right - w, r.left + r.width / 2 - w / 2));
    const above = r.top - h - 12 > 56;   // keep clear of the top bar
    pop.className = "term-pop " + (above ? "above" : "below");
    pop.style.left = `${left + scrollX}px`;
    pop.style.top = `${(above ? r.top - h - 10 : r.bottom + 10) + scrollY}px`;
    pop.style.setProperty("--arrow", `${r.left + r.width / 2 - left}px`);
    term.setAttribute("aria-expanded", "true");
    open = term;
  };
  document.addEventListener("click", (e) => {
    if (e.target.closest(".term-pop a[href^='#']")) return close();   // a link inside a card jumps away
    const term = e.target.closest(".term");
    if (term) { const same = term === open; close(); if (!same) show(term); return; }
    if (!pop.contains(e.target)) close();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") return close();
    const term = e.target.closest?.(".term");
    if (term && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); const same = term === open; close(); if (!same) show(term); }
  });
  addEventListener("resize", close);
}

// ===================================================================================== roadmap findings
// Section 6 repeats the roadmap of Section 4.5 (Figure 9). Each box of the copy opens a card with its
// finding, from #term-defs, through the same code as the term cards (setupTerms).
const RM_CARDS = {
  "Synthetic discs": "rm-discs", "Tuning": "rm-tune", "Main experiment, 3,000 steps": "rm-main3k",
  "More Gaussians; longer training": "rm-longer", "Re-tuning for full training": "rm-retune",
  "Main experiment, fully trained": "rm-full", "Still photographs": "rm-still", "Degree 3 and 7": "rm-degree",
  "Frames between": "rm-between", "Synthetic arcs (two versions)": "rm-arcs", "Confirmatory test": "rm-confirm",
};
function roadmapFindings() {
  const slot = $(".rm-findings"), src = $("figure.roadmap svg");
  if (!slot || !src) return;
  // the copy gets its own marker ids, so the two figures do not share ids
  slot.innerHTML = src.outerHTML.replaceAll('id="rm-arrow', 'id="rm9-arrow').replaceAll("url(#rm-arrow", "url(#rm9-arrow");
  const svg = $("svg", slot);
  svg.setAttribute("aria-label", "Roadmap of the experiments; each box opens the finding of that step");
  for (const t of svg.querySelectorAll("text.rm-name")) {
    const key = RM_CARDS[t.textContent], box = t.previousElementSibling;
    if (!key || box?.tagName !== "rect") continue;
    const hit = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    ["x", "y", "width", "height", "rx"].forEach((k) => hit.setAttribute(k, box.getAttribute(k)));
    Object.entries({ class: "term rm-hit", "data-term": key, tabindex: 0, role: "button", "aria-expanded": "false", "aria-label": t.textContent })
      .forEach(([k, v]) => hit.setAttribute(k, v));
    svg.appendChild(hit);   // on top of the box and its text
  }
}

// A link to something inside a folded box (details) opens the box first, so the target can be seen.
function openFoldsFor(id) {
  const el = id && document.getElementById(id);
  let d = el?.closest("details"), opened = false;
  while (d) { if (!d.open) { d.open = true; opened = true; } d = d.parentElement?.closest("details"); }
  return opened ? el : null;
}
document.addEventListener("click", (e) => {
  const a = e.target.closest?.('a[href^="#"]');
  if (a) openFoldsFor(decodeURIComponent(a.getAttribute("href").slice(1)));
});
addEventListener("hashchange", () => openFoldsFor(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView());
openFoldsFor(decodeURIComponent(location.hash.slice(1)))?.scrollIntoView();

roadmapFindings();
setupToc();
// ===================================================================================== roadmap
// The lines of the roadmap (Section 4.5) where a research question enters get a purple background,
// sized to the text once the fonts are in.
function highlightRoadmap() {
  for (const t of $$("figure.roadmap text.rm-hl")) {
    const b = t.getBBox();
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    Object.entries({ x: b.x - 5, y: b.y - 2, width: b.width + 10, height: b.height + 4, rx: 3, class: "rm-hl-bg" })
      .forEach(([k, v]) => r.setAttribute(k, v));
    t.before(r);
  }
}

setupTerms();
document.fonts.ready.then(highlightRoadmap);
$$(".w-volume").forEach(volumeWidget);
fillTables();
