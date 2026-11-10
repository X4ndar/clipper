// clipper web UI: no build step, no framework. Three views behind hash routes.

const $ = (sel, el = document) => el.querySelector(sel);
const view = $("#view");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const STAGE_LABEL = {
  fetch: "Downloading the stream", ingest: "Reading the video", transcribe: "Transcribing", quality: "Checking the transcript",
  signals: "Finding the loud moments", candidates: "Shortlisting moments", judge: "Judging each moment",
  snap: "Trimming to clean cuts", reframe: "Following faces", captions: "Writing captions", render: "Rendering clips",
  metadata: "Writing titles and tags", report: "Building the report", job: "Finishing",
};
const TYPE_LABEL = { stream: "Stream", challenge: "Challenge", podcast: "Podcast", interview: "Interview", talking_head: "Talking head", tutorial: "Tutorial", other: "Other" };
const SCORE_LABEL = { hook: "Hook", standalone_sense: "Stands alone", payoff: "Payoff", peak: "Peak", quotability: "Quotable", clean_ending: "Clean end" };

let meta = null;
let stream = null; // open EventSource
let pollTimer = null;

// ---------- helpers ----------
async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: opts.body && !(opts.body instanceof Blob) ? { "Content-Type": "application/json" } : undefined });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* keep status text */ }
    throw new Error(msg);
  }
  return r.json();
}
function toast(msg, bad = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "show" + (bad ? " bad" : "");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.className = ""), bad ? 6000 : 2600);
}
function clock(s) {
  if (s == null || isNaN(s)) return "–";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(x).padStart(2, "0")}` : `${m}:${String(x).padStart(2, "0")}`;
}
function long(s) {
  if (!s) return "–";
  const h = Math.floor(s / 3600), m = Math.round((s % 3600) / 60);
  return h ? `${h} h ${m} min` : `${Math.max(1, m)} min`;
}
function ago(ts) {
  if (!ts) return "";
  const d = Date.now() / 1000 - ts;
  if (d < 90) return "just now";
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  if (d < 86400) return `${Math.round(d / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
function bytes(n) {
  if (n > 1e9) return (n / 1e9).toFixed(1) + " GB";
  if (n > 1e6) return (n / 1e6).toFixed(0) + " MB";
  return Math.max(1, Math.round(n / 1e3)) + " KB";
}
const isUrl = (s) => /^https?:\/\//i.test(s.trim());
const money = (v) => (typeof v === "number" ? (v === 0 ? "$0" : "$" + v.toFixed(v < 1 ? 3 : 2)) : "–");

function closeStream() {
  if (stream) { stream.close(); stream = null; }
  clearTimeout(pollTimer);
}

// ---------- modal: confirm, file picker ----------
const modal = $("#modal");
function closeModal() { modal.hidden = true; modal.innerHTML = ""; if (closeModal.back) closeModal.back.focus(); }
modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !modal.hidden) closeModal(); });

function confirmBox(title, text, action, danger = true) {
  return new Promise((resolve) => {
    closeModal.back = document.activeElement;
    modal.innerHTML = `<div class="sheet" role="dialog" aria-modal="true" aria-label="${esc(title)}">
      <header><h3>${esc(title)}</h3></header>
      <div class="confirm"><p>${esc(text)}</p>
      <div class="actions"><button class="btn" data-x>Cancel</button><button class="btn ${danger ? "danger" : ""}" data-ok>${esc(action)}</button></div></div></div>`;
    modal.hidden = false;
    $("[data-x]", modal).onclick = () => { closeModal(); resolve(false); };
    $("[data-ok]", modal).onclick = () => { closeModal(); resolve(true); };
    $("[data-x]", modal).focus();
  });
}

function pickFile(kind) {
  return new Promise((resolve) => {
    closeModal.back = document.activeElement;
    modal.hidden = false;
    const go = async (path) => {
      let d;
      try { d = await api(`/api/fs?kind=${kind}&path=${encodeURIComponent(path)}`); }
      catch (e) { toast(e.message, true); return; }
      const rows = [];
      if (d.parent !== null) rows.push(`<li><button data-dir="${esc(d.parent)}"><span class="k">↑</span>Up one folder</button></li>`);
      for (const x of d.dirs) rows.push(`<li><button data-dir="${esc(x.path)}"><span class="k">▸</span>${esc(x.name)}</button></li>`);
      for (const x of d.files) rows.push(`<li class="file"><button data-file="${esc(x.path)}"><span class="k">●</span>${esc(x.name)}<span class="size">${bytes(x.size)}</span></button></li>`);
      if (d.path && !d.files.length && !d.dirs.length) rows.push(`<li><button disabled class="dim">Nothing to pick in this folder.</button></li>`);
      modal.innerHTML = `<div class="sheet" role="dialog" aria-modal="true" aria-label="Choose a file">
        <header><h3>${kind === "video" ? "Choose a video" : "Choose a chat file"}</h3><button class="btn quiet" data-x>Close</button></header>
        <div class="path">${esc(d.path || "Folders on this computer")}</div><ul>${rows.join("")}</ul></div>`;
      $("[data-x]", modal).onclick = () => { closeModal(); resolve(null); };
      modal.querySelectorAll("[data-dir]").forEach((b) => (b.onclick = () => go(b.dataset.dir)));
      modal.querySelectorAll("[data-file]").forEach((b) => (b.onclick = () => { closeModal(); resolve(b.dataset.file); }));
      (modal.querySelector("li button") || $("[data-x]", modal)).focus();
    };
    go("");
  });
}

// ---------- top bar ----------
async function loadMeta() {
  meta = await api("/api/meta");
  const u = meta.usage;
  $("#usage").innerHTML = `transcription credit left <b>$${u.asr_credit_left_usd.toFixed(2)}</b> · AI spend <b>${money(u.llm_cost_usd)}</b>`;
  return meta;
}
function setNav(name) {
  document.querySelectorAll("[data-nav]").forEach((a) => (a.dataset.nav === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
}

// ---------- home ----------
function composerHtml() {
  const d = meta.defaults;
  const backends = [
    ["claude_code", "Claude subscription", meta.claude_cli],
    ["anthropic", "Claude API key", meta.keys.anthropic],
    ["fake", "Offline test (no AI)", true],
  ];
  const def = backends.find((b) => b[0] === d.llm_backend && b[2]) || backends.find((b) => b[2]);
  return `<form class="composer" id="composer" novalidate>
    <div class="source">
      <input id="source" name="source" autocomplete="off" spellcheck="false" aria-label="Stream link or video file"
             placeholder="Paste a Twitch or Kick stream link, or drop a video here">
      <button type="button" class="btn" id="browse">Choose a file</button>
    </div>
    <div class="chosen" id="chosen" hidden></div>
    <div class="options" id="options" hidden>
      <div class="row wide"><div class="field" role="group" aria-label="Kind of video"><span class="note">Kind of video</span>
        <div class="seg">${meta.content_types.map((t) => `<label><input type="radio" name="content_type" value="${t}"><span>${TYPE_LABEL[t]}</span></label>`).join("")}</div></div></div>
      <div class="row wide"><label class="field">What should it look for? (optional)
        <input class="input" name="direction" placeholder="e.g. funny fails and clutch wins" maxlength="300"></label></div>
      <div class="row">
        <label class="field">Clips to make, at most<input class="input" type="number" name="max_clips" min="1" max="30" value="${d.max_clips}"></label>
        <label class="field">Shortest clip, seconds<input class="input" type="number" name="min_duration" min="5" max="170" value="${d.min_duration}"></label>
        <label class="field">Longest clip, seconds<input class="input" type="number" name="max_duration" min="10" max="${d.hard_cap_duration}" value="${d.max_duration}"></label>
        <label class="field">AI that picks the moments<select class="input" name="llm_backend">
          ${backends.map((b) => `<option value="${b[0]}" ${b === def ? "selected" : ""} ${b[2] ? "" : "disabled"}>${b[1]}${b[2] ? "" : " (not set up)"}</option>`).join("")}</select></label>
      </div>
      <div class="row wide" id="chatrow"><label class="field">Chat file (optional, makes stream picks sharper)
        <span class="inline-pick"><input class="input" name="chat" placeholder="TwitchDownloader JSON, or [HH:MM:SS] user: message lines" autocomplete="off" spellcheck="false">
        <button type="button" class="btn" id="browsechat">Choose</button></span></label></div>
      <p class="note" id="linknote" hidden></p>
      <div class="go">
        <label class="check"><input type="checkbox" name="rights" required>
          <span>This video is mine, or I have the rights to clip it.</span></label>
        <button class="btn primary" id="make">Make clips</button>
      </div>
    </div>
  </form>`;
}

function jobRow(j) {
  const thumbs = j.posters.length
    ? j.posters.map((r) => `<img loading="lazy" alt="" src="/media/${encodeURIComponent(j.id)}/poster/${r}">`).join("")
    : "<i></i><i></i><i></i>";
  const n = j.status === "done" ? `<div class="count">${j.clips ?? 0}<small>${j.clips === 1 ? "clip" : "clips"}</small></div>`
    : `<span class="pill ${j.status}">${j.status === "failed" ? "stopped on an error" : j.status}</span>`;
  return `<a class="job" href="#/job/${encodeURIComponent(j.id)}"><div class="thumbs">${thumbs}</div>
    <div><div class="name">${esc(j.title)}</div><div class="meta"><span>${TYPE_LABEL[j.content_type] || ""}</span>
    <span>${long(j.duration)}</span>${j.has_chat ? "<span>with chat</span>" : ""}<span>${ago(j.created_at)}</span></div></div>${n}</a>`;
}
function pendingRow(t) {
  return `<a class="job" href="#/task/${t.key}"><div class="thumbs"><i></i><i></i><i></i></div>
    <div><div class="name">${esc(t.title)}</div><div class="meta"><span class="mono">${esc(t.source || "")}</span></div></div>
    <span class="pill ${t.status === "failed" ? "failed" : t.status === "queued" || t.status === "cancelled" ? t.status : "running"}">${t.status === "fetching" ? "downloading" : t.status}</span></a>`;
}

async function home() {
  setNav("home");
  await loadMeta();
  const missing = [];
  if (!meta.ffmpeg) missing.push("ffmpeg was not found, so nothing can be rendered. Install it and restart clipper.");
  if (!meta.keys.assemblyai) missing.push("No AssemblyAI key found. Add ASSEMBLYAI_API_KEY to the .env file to transcribe.");
  view.innerHTML = `<h1>Cut the stream<br>down to its best minutes</h1>
    <p class="lead">Give it a finished stream. It finds the moments worth keeping, crops them for phones, and adds captions.</p>
    ${missing.map((m) => `<p class="note warn">${esc(m)}</p>`).join("")}
    ${composerHtml()}
    <h2>Jobs <small id="jobcount"></small></h2><div class="jobs" id="jobs"><div class="empty">Loading…</div></div>`;
  wireComposer();
  const refresh = async () => {
    let d;
    try { d = await api("/api/jobs"); } catch { return; }
    if (!$("#jobs")) return;
    const rows = d.pending.map(pendingRow).concat(d.jobs.map(jobRow));
    $("#jobs").innerHTML = rows.join("") || `<div class="empty">No jobs yet. Paste a stream link above to make the first one.</div>`;
    $("#jobcount").textContent = d.jobs.length ? String(d.jobs.length) : "";
    if (d.pending.some((t) => ["queued", "fetching", "running"].includes(t.status)) || d.jobs.some((j) => j.status === "running")) {
      pollTimer = setTimeout(refresh, 3000);
    }
  };
  refresh();
}

function wireComposer() {
  const form = $("#composer"), src = $("#source"), opts = $("#options"), chosen = $("#chosen");
  const setType = (t) => { const r = form.querySelector(`[name=content_type][value=${t}]`); if (r) r.checked = true; };
  setType("stream");
  const update = () => {
    const v = src.value.trim();
    opts.hidden = !v;
    const url = isUrl(v);
    $("#chatrow").hidden = url;
    const note = $("#linknote");
    note.hidden = !url;
    if (url) {
      note.className = "note" + (meta.fetch ? "" : " warn");
      note.textContent = meta.fetch
        ? "The stream is downloaded first (up to 1080p), with its chat replay when the site still has it. Live streams are refused: wait for the stream to end."
        : "Downloading from a link needs the fetch extra: uv sync --frozen --extra fetch --extra web";
    }
  };
  src.addEventListener("input", () => { chosen.hidden = true; update(); });
  $("#browse").onclick = async () => { const f = await pickFile("video"); if (f) { src.value = f; update(); } };
  $("#browsechat").onclick = async () => { const f = await pickFile("chat"); if (f) form.chat.value = f; };

  // drag and drop: the file is copied into clipper's uploads folder
  const over = (on) => form.classList.toggle("over", on);
  ["dragenter", "dragover"].forEach((ev) => form.addEventListener(ev, (e) => { e.preventDefault(); over(true); }));
  ["dragleave", "drop"].forEach((ev) => form.addEventListener(ev, (e) => { e.preventDefault(); over(false); }));
  form.addEventListener("drop", async (e) => {
    const text = e.dataTransfer.getData("text/uri-list") || e.dataTransfer.getData("text/plain");
    const file = e.dataTransfer.files[0];
    if (!file) { if (text && isUrl(text)) { src.value = text.trim(); update(); } return; }
    chosen.hidden = false;
    chosen.innerHTML = `Copying <b>${esc(file.name)}</b> (${bytes(file.size)}) into clipper…`;
    try {
      const r = await api(`/api/upload?name=${encodeURIComponent(file.name)}`, { method: "PUT", body: file });
      if (/\.(json|txt|log)$/i.test(file.name)) { form.chat.value = r.path; chosen.innerHTML = `Chat file added: <b>${esc(file.name)}</b>`; }
      else { src.value = r.path; chosen.innerHTML = `Ready: <b>${esc(file.name)}</b>`; }
      update();
    } catch (err) { chosen.hidden = true; toast(err.message, true); }
  });

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const f = new FormData(form);
    const body = Object.fromEntries(f.entries());
    body.rights = form.rights.checked;
    for (const k of ["max_clips", "min_duration", "max_duration"]) body[k] = Number(body[k]);
    if (!body.rights) { toast("Tick the box to confirm the video is yours to clip.", true); form.rights.focus(); return; }
    const btn = $("#make");
    btn.disabled = true;
    try {
      const r = await api("/api/jobs", { method: "POST", body: JSON.stringify(body) });
      location.hash = `#/task/${r.task}`;
    } catch (err) { toast(err.message, true); btn.disabled = false; }
  });
  update();
}

// ---------- progress rail (shared by task and job views) ----------
function railHtml(stages, withFetch) {
  const names = (withFetch ? ["fetch"] : []).concat(meta.stages);
  return `<section class="rail" id="rail" aria-live="polite">
    <div class="now" id="now">Waiting in line</div><div class="line" id="line"></div>
    <div class="ticks">${names.map((s) => `<i data-stage="${s}" title="${esc(STAGE_LABEL[s])}"></i>`).join("")}</div>
    <div class="ticklabels"><span>${withFetch ? "download" : "read"}</span><span>transcribe</span><span>pick moments</span><span>crop and caption</span><span>render</span></div>
    <div class="actions"><button class="btn danger" id="stop">Stop</button></div></section>`;
}
function applyEvent(ev) {
  const tick = (s) => document.querySelector(`.ticks [data-stage="${s}"]`);
  const now = $("#now"), line = $("#line");
  if (!now) return;
  if (ev.stage === "job") return;
  const el = tick(ev.stage);
  if (ev.status === "start") {
    document.querySelectorAll(".ticks .run").forEach((x) => x.classList.replace("run", "done"));
    if (el) el.className = "run";
    now.textContent = STAGE_LABEL[ev.stage] || ev.stage;
    line.textContent = "";
  } else if (ev.status === "done" || ev.status === "skip") {
    if (el) el.className = "done";
  } else if (ev.status === "error") {
    if (el) el.className = "failed";
    line.textContent = ev.msg;
  } else if (ev.msg) {
    line.textContent = ev.msg;
  }
}
function follow(taskKey, events, onEnd, onCreated) {
  closeStream();
  let last = -1;
  for (const ev of events || []) { applyEvent(ev); last = ev.seq; }
  stream = new EventSource(`/api/tasks/${taskKey}/events?after=${last}`);
  stream.onmessage = (m) => {
    const ev = JSON.parse(m.data);
    applyEvent(ev);
    if (ev.stage === "job" && ev.status === "created" && onCreated) onCreated(ev.msg);
  };
  stream.addEventListener("end", () => { closeStream(); onEnd(); });
  stream.onerror = () => { /* the browser reconnects by itself */ };
  const stop = $("#stop");
  if (stop) stop.onclick = async () => {
    stop.disabled = true;
    await api(`/api/tasks/${taskKey}/cancel`, { method: "POST" }).catch((e) => toast(e.message, true));
    $("#line").textContent = "Stopping after the current step…";
  };
}

// ---------- task view: before the job folder exists (queued, downloading) ----------
async function taskView(key) {
  setNav("home");
  if (!meta) await loadMeta();
  let t;
  try { t = await api(`/api/tasks/${key}`); } catch { location.hash = "#/"; return; }
  if (t.job_id) { location.replace(`#/job/${encodeURIComponent(t.job_id)}`); return; }
  const url = isUrl(t.source || "");
  view.innerHTML = `<div class="crumbs"><a href="#/">Jobs</a> /</div><h1>${esc(t.title)}</h1>
    <div class="facts"><span class="mono">${esc(t.source || "")}</span></div>
    ${t.status === "failed" ? `<div class="problem"><h3>${url ? "The download did not work" : "The job could not start"}</h3><pre>${esc(t.error)}</pre>
        <a class="btn" href="#/">Back to jobs</a></div>`
      : t.status === "cancelled" ? `<div class="problem"><h3>Stopped</h3><pre>You stopped this before it finished. What was downloaded is kept, so starting again picks up from there.</pre><a class="btn" href="#/">Back to jobs</a></div>`
      : railHtml([], url)}`;
  if (["failed", "cancelled"].includes(t.status)) return;
  follow(key, t.events, () => taskView(key), (jobId) => { closeStream(); location.replace(`#/job/${encodeURIComponent(jobId)}`); });
}

// ---------- job view ----------
function stripHtml(job) {
  const sig = job.signals, total = job.duration || (sig && sig.seconds) || 0;
  if (!sig || !total) return "";
  const area = (xs) => {
    const n = xs.length, pts = xs.map((v, i) => `${((i + 0.5) / n * 1000).toFixed(1)},${(100 - Math.min(1, v / 4.5) * 92).toFixed(1)}`);
    return `M0,100 L${pts.join(" L")} L1000,100 Z`;
  };
  const line = (xs) => {
    const n = xs.length;
    return "M" + xs.map((v, i) => `${((i + 0.5) / n * 1000).toFixed(1)},${(100 - Math.min(1, v / 4.5) * 92).toFixed(1)}`).join(" L");
  };
  const pct = (t) => (Math.max(0, Math.min(total, t)) / total * 100).toFixed(3) + "%";
  const cuts = job.clips.map((c) => `<button class="cut" data-rank="${c.rank}" style="left:${pct(c.start)};width:${((c.end - c.start) / total * 100).toFixed(3)}%"
      aria-label="Clip ${c.rank}, ${clock(c.start)} to ${clock(c.end)}"><b>${c.rank}</b></button>`).join("");
  const rejects = job.candidates.filter((c) => c.status !== "selected" && c.start != null)
    .map((c) => `<i class="cut reject" style="left:${pct(c.start)};width:${((c.end - c.start) / total * 100).toFixed(3)}%"></i>`).join("");
  const peaks = (sig.peaks || []).map((p) => `<i class="peak" style="left:${pct(p.t)}"></i>`).join("");
  const stepS = [30, 60, 120, 300, 600, 900, 1800, 3600].find((s) => total / s <= 12) || 7200;
  let ticks = "";
  for (let t = 0; t <= total; t += stepS) ticks += `<span style="left:${pct(t)}"><em>${clock(t)}</em></span>`;
  return `<section class="strip" aria-label="Timeline of the whole video">
    <div class="legend"><span><i style="background:rgba(237,243,238,.4)"></i>how loud it got, against the minutes around it</span>
      ${sig.has_chat ? `<span><i style="background:var(--chat)"></i>chat activity (${sig.chat_messages.toLocaleString()} messages)</span>` : ""}
      <span><i style="background:var(--mark)"></i>clips cut</span>
      ${rejects ? `<span><i style="background:var(--alert);height:3px;vertical-align:middle"></i>considered, not kept</span>` : ""}</div>
    <div class="band draw" id="band"><svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">
      <path class="audio" d="${area(sig.audio || [])}"/>${sig.chat ? `<path class="chatline" d="${line(sig.chat)}"/>` : ""}</svg>
      ${peaks}${rejects}${cuts}</div>
    <div class="ruler" aria-hidden="true">${ticks}</div><div class="readout" id="readout"></div></section>`;
}

function clipCard(job, c) {
  const id = encodeURIComponent(job.id);
  const bars = Object.keys(SCORE_LABEL).filter((k) => k in (c.scores || {}))
    .map((k) => `<span>${SCORE_LABEL[k]}</span><i style="--v:${c.scores[k] * 10}%"></i><span class="mono">${Math.round(c.scores[k])}</span>`).join("");
  const found = c.source === "manual" ? "picked by hand" : c.source === "peaks" ? "found by a loud moment" : "found in the transcript";
  const flags = (c.needs_review ? c.review_reasons : []).map((r) => `<span class="flag warn">${esc(r)}</span>`).join("");
  const pieces = c.segments && c.segments.length > 1 ? ` · ${c.segments.length} pieces` : "";
  return `<article class="clip" id="clip-${c.rank}" data-rank="${c.rank}">
    <div class="frame"><span class="rank">${String(c.rank).padStart(2, "0")}</span>
      ${c.has_video ? `<img loading="lazy" alt="" src="/media/${id}/poster/${c.rank}">
        <button class="play" data-play="${c.rank}" aria-label="Play clip ${c.rank}: ${esc(c.title)}"><svg viewBox="0 0 48 48"><path d="M14 8l26 16-26 16z"/></svg></button>
        <span class="len">${Math.round(c.duration)} s</span>` : ""}</div>
    <div class="body"><h3>${esc(c.title || "Untitled clip")}</h3>
      <div class="when">${clock(c.start)} – ${clock(c.end)}${pieces} · ${found}</div>
      ${c.hook_text ? `<div class="hook">Opens with “${esc(c.hook_text)}”</div>` : ""}
      ${flags ? `<div class="flags">${flags}</div>` : ""}
      ${bars ? `<div class="bars">${bars}</div>` : ""}
      <details><summary>Why this one, and what is said</summary><p>${esc(c.rationale || "")}</p><p>${esc(c.text || "")}</p>
        ${c.tags && c.tags.length ? `<p class="mono">${c.tags.map((t) => "#" + esc(t)).join(" ")}</p>` : ""}</details></div>
    <div class="foot">${c.has_video ? `<a class="btn" href="/media/${id}/clip/${c.rank}?download=1" download>Download</a>` : ""}</div></article>`;
}

function candidateRows(job) {
  const order = ["selected", "over_quota", "duplicate", "below_threshold", "out_of_range", "rejected", "error", "skipped"];
  const cs = [...job.candidates].sort((a, b) => (order.indexOf(a.status) + 99) % 99 - (order.indexOf(b.status) + 99) % 99 || (b.editorial || 0) - (a.editorial || 0));
  return cs.map((c) => `<tr><td><span class="flag ${c.status === "selected" ? "sel" : ["over_quota", "duplicate"].includes(c.status) ? "near" : ""}">${esc(c.label)}</span></td>
    <td class="mono">${c.start != null ? `${clock(c.start)}–${clock(c.end)}` : "–"}<br><span class="dim">${c.source === "peaks" ? "loud moment" : c.source === "manual" ? "by hand" : "transcript"}</span></td>
    <td class="mono">${c.editorial != null ? c.editorial.toFixed(1) : "–"}</td>
    <td>${esc(c.why || c.pass1 || "")}</td><td class="dim">${esc(c.opening)}</td></tr>`).join("");
}

async function jobView(id) {
  setNav("home");
  if (!meta) await loadMeta();
  let job;
  try { job = await api(`/api/jobs/${encodeURIComponent(id)}`); }
  catch (e) { view.innerHTML = `<div class="crumbs"><a href="#/">Jobs</a> /</div><h1>Not found</h1><p class="lead">${esc(e.message)}</p>`; return; }
  const st = job.stats || {}, running = job.status === "running";
  const failed = job.stages.find((s) => s.status === "failed");
  const eid = encodeURIComponent(job.id);
  const facts = [
    `<span><b>${TYPE_LABEL[job.content_type] || job.content_type}</b></span>`, `<span><b>${long(job.duration)}</b> long</span>`,
    job.clips.length || job.status === "done" ? `<span><b>${job.clips.length}</b> ${job.clips.length === 1 ? "clip" : "clips"} from <b>${job.candidates.length}</b> moments considered</span>` : "",
    job.signals && job.signals.has_chat ? `<span>chat used</span>` : "",
    st.total_cost_usd != null ? `<span>cost <b>${money(st.total_cost_usd)}</b></span>` : "",
    job.spec.direction ? `<span>looking for <b>“${esc(job.spec.direction)}”</b></span>` : "",
    job.selection_source ? `<span><b>${esc(job.selection_source)}</b></span>` : "",
  ].filter(Boolean).join("");

  let problem = "";
  if (failed && !running) {
    const gate = failed.name === "quality";
    problem = `<div class="problem"><h3>${gate ? "The transcript looks unreliable" : `Stopped while ${(STAGE_LABEL[failed.name] || failed.name).toLowerCase()}`}</h3>
      <pre>${esc(failed.error || "")}</pre>
      ${gate ? `<p>This check stops the job before any AI is used. Fast speech over music often trips it. You can continue, and review the clips with more care.</p>` : ""}
      <div class="actions" style="margin:0">${gate ? `<button class="btn" data-resume="quality" data-ignore="1">Continue anyway</button>` : `<button class="btn" data-resume="${failed.name}">Try this step again</button>`}</div></div>`;
  } else if (!running && job.status === "stopped") {
    problem = `<div class="problem"><h3>Stopped before the end</h3><pre>Finished steps are kept.</pre><div class="actions" style="margin:0"><button class="btn" data-resume="">Continue</button></div></div>`;
  }
  const warns = (job.warnings || []).filter((w) => !(job.clips.length && w.startsWith("No clip passed")));
  view.innerHTML = `<div class="crumbs"><a href="#/">Jobs</a> /</div><h1>${esc(job.title)}</h1>
    <div class="facts">${facts}</div>
    ${running ? railHtml(job.stages, false) : problem}
    ${stripHtml(job)}
    ${job.clips.length ? `<h2>Clips <small>ranked best first · review each one before posting</small></h2><div class="clips">${job.clips.map((c) => clipCard(job, c)).join("")}</div>`
      : job.status === "done" ? `<h2>Clips</h2><div class="empty">Nothing passed the bar this time. That is a valid result: the table below says why each moment was left out. A short note under “What should it look for?” often helps on a second run.</div>` : ""}
    ${warns.length ? `<details class="warnings"><summary>${warns.length} ${warns.length === 1 ? "note" : "notes"} from this run</summary><ul>${warns.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></details>` : ""}
    ${job.candidates.length ? `<h2>Every moment it considered <small>${job.candidates.length}</small></h2>
      <div class="tablewrap"><table><thead><tr><th>Result</th><th>Where</th><th>Score</th><th>Why</th><th>Opening words</th></tr></thead><tbody>${candidateRows(job)}</tbody></table></div>` : ""}
    ${running ? "" : `<div class="actions" style="margin-top:44px">
      <button class="btn" id="openfolder">Open the clips folder</button>
      ${job.status === "done" ? `<a class="btn" href="/media/${eid}/report" target="_blank" rel="noopener">Full report</a>` : ""}
      <button class="btn" data-resume="candidates">Pick the moments again</button>
      <button class="btn" data-resume="reframe">Render again</button>
      <button class="btn quiet danger" id="delete">Delete this job</button></div>`}`;

  // rail state from saved stages, then live events
  if (running) {
    for (const s of job.stages) { const el = document.querySelector(`.ticks [data-stage="${s.name}"]`); if (el && s.status === "done") el.className = "done"; }
    follow(job.task, job.events, () => jobView(id));
  }
  // strip <-> cards
  const band = $("#band");
  const mark = (rank, on) => {
    document.querySelectorAll(`.cut[data-rank="${rank}"], .clip[data-rank="${rank}"]`).forEach((el) => el.classList.toggle("on", on));
  };
  if (band) {
    const total = job.duration || job.signals.seconds;
    band.addEventListener("mousemove", (e) => {
      const r = band.getBoundingClientRect(), t = (e.clientX - r.left) / r.width * total;
      const c = job.clips.find((c) => t >= c.start && t <= c.end);
      $("#readout").textContent = clock(t) + (c ? `  ·  clip ${c.rank}: ${c.title || ""}` : "");
    });
    band.addEventListener("mouseleave", () => ($("#readout").textContent = ""));
    band.querySelectorAll(".cut[data-rank]").forEach((b) => {
      b.onmouseenter = () => mark(b.dataset.rank, true);
      b.onmouseleave = () => mark(b.dataset.rank, false);
      b.onclick = () => { const card = $(`#clip-${b.dataset.rank}`); card.scrollIntoView({ behavior: "smooth", block: "center" }); const p = $("[data-play]", card); if (p) p.focus({ preventScroll: true }); };
    });
  }
  document.querySelectorAll(".clip").forEach((card) => {
    card.onmouseenter = () => mark(card.dataset.rank, true);
    card.onmouseleave = () => mark(card.dataset.rank, false);
  });
  view.querySelectorAll("[data-play]").forEach((b) => (b.onclick = () => {
    document.querySelectorAll(".frame video").forEach((v) => v.pause());
    const frame = b.parentElement, v = document.createElement("video");
    v.src = `/media/${eid}/clip/${b.dataset.play}`;
    v.controls = true; v.autoplay = true; v.playsInline = true;
    frame.querySelector("img")?.remove(); frame.querySelector(".len")?.remove(); b.remove();
    frame.appendChild(v);
    v.focus();
  }));
  view.querySelectorAll("[data-resume]").forEach((b) => (b.onclick = async () => {
    b.disabled = true;
    try {
      await api(`/api/jobs/${eid}/resume`, { method: "POST", body: JSON.stringify({ from_stage: b.dataset.resume || null, ignore_quality: !!b.dataset.ignore }) });
      jobView(id);
    } catch (e) { toast(e.message, true); b.disabled = false; }
  }));
  const of = $("#openfolder");
  if (of) of.onclick = () => api(`/api/jobs/${eid}/open`, { method: "POST" }).then((r) => toast(`Opened ${r.path}`)).catch((e) => toast(e.message, true));
  const del = $("#delete");
  if (del) del.onclick = async () => {
    if (!(await confirmBox("Delete this job?", `This removes the clips, transcript and report of “${job.title}”. The source video is not touched.`, "Delete job"))) return;
    try { await api(`/api/jobs/${eid}`, { method: "DELETE" }); toast("Job deleted"); location.hash = "#/"; }
    catch (e) { toast(e.message, true); }
  };
}

// ---------- settings ----------
async function settingsView() {
  setNav("settings");
  await loadMeta();
  const s = await api("/api/settings");
  const yn = (ok, yes, no) => `<span class="${ok ? "yes" : "no"}">${ok ? yes : no}</span>`;
  const sel = (name, value, options) => `<select class="input" name="${name}">${options.map(([v, l]) => `<option value="${v}" ${String(v) === String(value) ? "selected" : ""}>${l}</option>`).join("")}</select>`;
  const chk = (name, value, label) => `<label class="check"><input type="checkbox" name="${name}" ${value ? "checked" : ""}><span>${label}</span></label>`;
  view.innerHTML = `<h1>Settings</h1><p class="lead">These apply to new jobs. They are saved in <span class="mono">${esc(meta.home)}</span>.</p>
  <div class="panel"><h3>What is set up</h3><div class="status">
    <div><b>Transcription</b>${yn(meta.keys.assemblyai, "AssemblyAI key found", "No AssemblyAI key in .env")}</div>
    <div><b>Claude subscription</b>${yn(meta.claude_cli, "Claude Code is installed", "Claude Code not found")}</div>
    <div><b>Claude API key</b>${yn(meta.keys.anthropic, "Key found", "No key in .env (optional)")}</div>
    <div><b>Stream download</b>${yn(meta.fetch, "Ready", "Install the fetch extra")}</div>
    <div><b>ffmpeg</b>${yn(meta.ffmpeg, "Found", "Not found")}</div></div>
    <p class="note">Keys live in the .env file and are never shown here.</p></div>
  <form id="settings">
  <div class="panel"><h3>Picking moments</h3><div class="row">
    <label class="field">AI that picks the moments${sel("llm.backend", s.llm.backend, [["claude_code", "Claude subscription"], ["anthropic", "Claude API key"], ["fake", "Offline test (no AI)"]])}</label>
    <label class="field">How moments are found${sel("selection.mode", s.selection.mode, [["auto", "Automatic"], ["both", "Transcript and loud moments"], ["peaks", "Loud moments only"], ["transcript", "Transcript only"]])}</label>
    <label class="field">Clips per video, at most<input class="input" type="number" name="selection.max_clips" min="1" max="30" value="${s.selection.max_clips}"></label>
    <label class="field">Shortest clip, seconds<input class="input" type="number" name="selection.min_duration" min="5" value="${s.selection.min_duration}"></label>
    <label class="field">Longest clip, seconds<input class="input" type="number" name="selection.max_duration" min="10" value="${s.selection.max_duration}"></label></div>
    ${chk("quality.block_on_fail", s.quality.block_on_fail, "Stop before the AI step when the transcript looks unreliable")}</div>
  <div class="panel"><h3>Framing</h3><div class="row">
    <label class="field">Crop${sel("render.reframe", s.render.reframe, [["face", "Follow faces"], ["center", "Middle of the frame"]])}</label>
    <label class="field">Layout${sel("render.layout", s.render.layout, [["auto", "Automatic"], ["single", "One crop"], ["stack", "Camera on top, gameplay below"]])}</label></div>
    ${chk("reframe.follow_speaker", s.reframe.follow_speaker, "When people do not all fit, prefer the one who is talking")}</div>
  <div class="panel"><h3>Captions</h3>
    ${chk("captions.enabled", s.captions.enabled, "Burn captions into the clips")}
    <div class="row">
    <label class="field">Style${sel("captions.mode", s.captions.mode, [["word", "Highlight the spoken word"], ["karaoke", "Karaoke fill"], ["plain", "Plain"]])}</label>
    <label class="field">Size<input class="input" type="number" name="captions.size" min="30" max="160" value="${s.captions.size}"></label>
    <label class="field">Text colour<input class="input" type="color" name="captions.primary_color" value="${esc(s.captions.primary_color.slice(0, 7))}"></label>
    <label class="field">Highlight colour<input class="input" type="color" name="captions.highlight_color" value="${esc(s.captions.highlight_color.slice(0, 7))}"></label>
    <label class="field">Outline colour<input class="input" type="color" name="captions.outline_color" value="${esc(s.captions.outline_color.slice(0, 7))}"></label></div>
    ${chk("captions.uppercase", s.captions.uppercase, "All capitals")}
    ${chk("hook.enabled", s.hook.enabled, "Show a hook line over the first seconds")}</div>
  <div class="panel"><h3>Downloading streams</h3><div class="row">
    <label class="field">Highest quality to download${sel("fetch.max_height", s.fetch.max_height, [[1080, "1080p"], [720, "720p (smaller, faster)"], [1440, "1440p"], [2160, "4K"]])}</label></div>
    ${chk("fetch.chat", s.fetch.chat, "Also download the chat replay")}</div>
  <div class="actions"><button class="btn primary">Save settings</button></div></form>`;
  $("#settings").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = {};
    for (const el of e.target.elements) {
      if (!el.name) continue;
      const [sec, key] = el.name.split(".");
      const v = el.type === "checkbox" ? el.checked : el.type === "number" || el.name === "fetch.max_height" ? Number(el.value) : el.type === "color" ? el.value.toUpperCase() : el.value;
      (body[sec] ||= {})[key] = v;
    }
    try { await api("/api/settings", { method: "PUT", body: JSON.stringify(body) }); toast("Settings saved"); loadMeta(); }
    catch (err) { toast(err.message, true); }
  });
}

// ---------- router ----------
async function route() {
  closeStream();
  if (!modal.hidden) closeModal();
  const [, name, arg] = (location.hash || "#/").split("/");
  window.scrollTo(0, 0);
  try {
    if (name === "job" && arg) await jobView(decodeURIComponent(arg));
    else if (name === "task" && arg) await taskView(arg);
    else if (name === "settings") await settingsView();
    else await home();
  } catch (e) {
    view.innerHTML = `<h1>Something broke</h1><p class="lead">${esc(e.message)}</p><a class="btn" href="#/">Back to jobs</a>`;
  }
}
window.addEventListener("hashchange", route);
route();
