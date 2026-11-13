// Reconstruction videos (stg/export_videos.py). Each .recon widget plays one MP4 in which the original
// and the models are panels of the same frame, so they cannot drift apart. The widget picks the video
// for its clip and budget from media/recon/videos.json, and the wide (3 across) or narrow (2 across)
// layout by its own width. Nothing downloads until the reader plays or steps; one plays at a time.
(() => {
  const BASE = "media/recon/";
  const NARROW = 600;                       // widget width (px) below which the narrow layout is used
  let manifest;
  const getManifest = () => manifest ??= fetch(BASE + "videos.json", { cache: "no-cache" })
    .then((r) => { if (!r.ok) throw new Error(`videos.json: ${r.status}`); return r.json(); });
  const videos = [];

  // one playing at a time
  document.addEventListener("play", (e) => { for (const v of videos) if (v !== e.target) v.pause(); }, true);

  function setup(root) {
    const $ = (s) => root.querySelector(s);
    const video = $("video"), play = $(".r-play"), prev = $(".r-prev"), next = $(".r-next"), seek = $(".r-seek");
    const readout = $(".r-readout"), meta = $(".r-meta");
    const module = root.dataset.recon;
    const state = { clip: root.dataset.clip ?? null, budget: 6000 };
    let entry = null, layout = null, frame = 0, pending = null, entries = [];
    videos.push(video);

    const fps = () => entry.fps;
    const count = () => entry.frames.length;
    const current = () => Math.min(count() - 1, Math.max(0, Math.floor(video.currentTime * fps() + 1e-3)));

    function show(i) {
      frame = i;
      const f = entry.frames[i];
      seek.value = i;
      const kind = f.kind === "fitted" ? "Fitted frame" : "Unseen frame";
      readout.innerHTML = `Frame ${i + 1} of ${count()} · ${module === "arcs" ? "frame" : "original frame"} ${f.frame} · ` +
        `<span class="r-kind ${f.kind}">${kind}</span>`;
    }

    function describe() {
      const b = entry.budget.toLocaleString("en-US");
      const clip = module === "arcs" ? "synthetic arcs 0° and 360°" : entry.clip;
      const rate = `${(fps() * video.playbackRate).toLocaleString("en-US")} frames per second`;
      meta.textContent = `Clip: ${clip} · start frame ${entry.start} · budget ${b} stored scalars · ` +
        `${entry.iterations.toLocaleString("en-US")} training steps · stride ${entry.stride} · seed ${entry.seed} · ` +
        `playing at ${rate}`;
    }

    function goTo(i) {
      i = (i + count()) % count();
      show(i);
      if (video.readyState >= 1) { video.currentTime = (i + 0.5) / fps(); return; }
      pending = i;                          // applied when the metadata arrives
      if (video.preload !== "auto") { video.preload = "auto"; video.load(); }
    }

    function choose(keepPlaying) {
      const want = entries.find((e) => (module === "arcs" || e.clip === state.clip) && e.budget === state.budget);
      if (!want) { readout.textContent = "This video is not available."; return; }
      const lay = root.clientWidth < NARROW ? "narrow" : "wide";
      if (want === entry && lay === layout) return;
      const wasPlaying = keepPlaying && !video.paused;
      const keep = entry ? frame : 0;
      entry = want; layout = lay;
      const file = entry.layouts[layout];
      video.width = file.width; video.height = file.height;
      video.style.aspectRatio = `${file.width} / ${file.height}`;
      video.poster = BASE + file.poster;
      seek.max = count() - 1;
      const loaded = video.readyState >= 1 || !video.paused || video.preload === "auto";
      video.preload = loaded ? "auto" : "none";
      video.src = BASE + file.file;
      pending = loaded ? Math.min(keep, count() - 1) : null;
      show(Math.min(keep, count() - 1));
      describe();
      if (wasPlaying) video.play().catch(() => {});
    }

    video.addEventListener("loadedmetadata", () => {
      if (pending !== null) { const i = pending; pending = null; video.currentTime = (i + 0.5) / fps(); show(i); }
    });
    const track = () => { if (entry && pending === null) show(current()); };
    video.addEventListener("timeupdate", track);
    video.addEventListener("seeked", track);
    if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
      const tick = () => { track(); video.requestVideoFrameCallback(tick); };
      video.requestVideoFrameCallback(tick);
    }
    video.addEventListener("play", () => { play.textContent = "Pause"; });
    video.addEventListener("pause", () => { play.textContent = "Play"; });
    video.addEventListener("ratechange", () => entry && describe());
    video.addEventListener("error", () => { readout.textContent = "The video could not be played in this browser."; });

    play.addEventListener("click", () => { if (video.paused) video.play().catch(() => {}); else video.pause(); });
    prev.addEventListener("click", () => { video.pause(); goTo(frame - 1); });
    next.addEventListener("click", () => { video.pause(); goTo(frame + 1); });
    seek.addEventListener("input", () => { video.pause(); goTo(Number(seek.value)); });
    video.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault(); video.pause(); goTo(frame + (e.key === "ArrowLeft" ? -1 : 1));
      } else if (e.key === " ") { e.preventDefault(); play.click(); }
    });
    video.addEventListener("click", () => play.click());

    root.querySelectorAll(".seg[data-key]").forEach((group) => group.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      group.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      const key = group.dataset.key;
      state[key] = key === "budget" ? Number(b.dataset.value) : b.dataset.value;
      choose(true);
    }));
    $(".r-speed").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      $(".r-speed").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      video.playbackRate = Number(b.dataset.rate);
      video.defaultPlaybackRate = video.playbackRate;
    });
    let resize;
    new ResizeObserver(() => { clearTimeout(resize); resize = setTimeout(() => entry && choose(true), 150); }).observe(root);

    getManifest().then((m) => {
      entries = m[module] ?? [];
      choose(false);
      for (const el of [play, prev, next, seek]) el.disabled = false;
    }).catch(() => { readout.textContent = "The videos could not be loaded."; });
  }

  const start = () => document.querySelectorAll(".widget.recon").forEach(setup);
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
