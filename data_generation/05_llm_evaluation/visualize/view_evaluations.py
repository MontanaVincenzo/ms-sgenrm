"""Build a self-contained HTML viewer for the final (majority-voted) evaluations.

Reads the evaluated dataset (`single_pipeline.py` output, default
`src/data/dataset_eval.jsonl`) and emits one HTML file: the shared input request
(topic, text, expected voice style, audio) on top, then two columns
(pipeline1 / pipeline2) cascading ASR -> LLM -> TTS -> the selected evaluation
-> the majority-vote summary. Navigate with the arrows, the id box, the
left/right keys, or the filter. Audio is embedded as base64 so the file works by
double-clicking -- no server.

    python src/llm_evaluation/visualize/view_evaluations.py
    python src/llm_evaluation/visualize/view_evaluations.py --limit 40 --link-audio
"""

import argparse
import base64
import json
import mimetypes
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]          # .../synthetic_data_generation
DATA_DIR = REPO_ROOT / "src" / "data"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", type=Path, default=DATA_DIR / "dataset_eval.jsonl",
                   help="evaluated dataset with pipeline*.overall_evaluation")
    p.add_argument("--out", type=Path, default=HERE / "evaluations.html")
    p.add_argument("--audio-dir", type=Path, default=None,
                   help="override dir for the input-request audio basenames")
    p.add_argument("--tts-audio-base", type=Path, default=REPO_ROOT / "src",
                   help="base dir the relative pipeline*.tts.output paths resolve against")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-audio", action="store_true", help="skip audio entirely (smallest file)")
    p.add_argument("--link-audio", action="store_true",
                   help="symlink wavs into <out>_audio/ and reference them relatively instead of "
                        "embedding (keeps the html small; needs `python -m http.server` to view)")
    return p.parse_args()


def _resolve(raw_path: str, base: Path | None, name_dir: Path | None) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    if name_dir is not None:
        p = name_dir / p.name
    elif base is not None and not p.is_absolute():
        p = base / p
    if p.is_file():
        return p
    if base is not None:  # last resort: search by basename under base
        return next(base.rglob(Path(raw_path).name), None)
    return None


def audio_data_uri(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def link_audio(src: Path | None, link_dir: Path) -> str | None:
    if src is None or not src.is_file():
        return None
    link_dir.mkdir(parents=True, exist_ok=True)
    dst = link_dir / src.name
    if not dst.exists():
        dst.symlink_to(src.resolve())
    return f"{link_dir.name}/{src.name}"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tts_effective(sample: dict, pk: str) -> tuple[str, str, str | None]:
    """Text the TTS actually spoke, the voice style it delivered, and the impact note.

    Mirrors `single_pipeline._tts_effective` / `tts_stage.build_inputs`.
    """
    stage = sample[pk]
    answer = stage["llm"].get("output") or ""
    expected = sample["additional_info"]["ideal_voice_response"]
    pert = stage["tts"]["perturbation"]
    ptype = pert.get("type")
    info = pert.get("additional_info")
    info = info if isinstance(info, dict) else {}

    spoken, delivered = answer, expected
    if ptype == "wrong_emotion":
        delivered = info.get("steered_style_instruction") or expected
    elif ptype in ("added_words", "missing_words"):
        spoken = info.get("perturbed_text") or answer
    return spoken, delivered, info.get("steering_impact")


def _intent(stage: dict) -> str | None:
    info = stage["llm"]["perturbation"].get("additional_info")
    if isinstance(info, dict):
        return info.get("intent")
    return info if isinstance(info, str) else None


def build_pipeline(sample: dict, pk: str, audio_uri) -> dict | None:
    stage = sample.get(pk)
    if not stage:
        return None
    spoken, delivered, impact = tts_effective(sample, pk)
    return {
        "asr_perturbation": stage["asr"]["perturbation"].get("type"),
        "asr_text": stage["asr"].get("output"),
        "llm_perturbation": stage["llm"]["perturbation"].get("type"),
        "llm_answer": stage["llm"].get("output"),
        "llm_intent": _intent(stage),
        "tts_perturbation": stage["tts"]["perturbation"].get("type"),
        "tts_impact": impact,
        "spoken_text": spoken,
        "delivered_voice_style": delivered,
        "tts_audio": audio_uri,
        "eval": stage.get("overall_evaluation"),
    }


def build_records(args) -> list[dict]:
    rows = _read_jsonl(args.eval)
    if args.limit is not None:
        rows = rows[: args.limit]
    link_dir = args.out.with_name(args.out.stem + "_audio")

    def audio_for(raw_path, *, is_tts):
        if args.no_audio:
            return None
        src = _resolve(
            raw_path,
            base=args.tts_audio_base if is_tts else None,
            name_dir=None if is_tts else args.audio_dir,
        )
        return link_audio(src, link_dir) if args.link_audio else audio_data_uri(src)

    n_in, n_tts = 0, 0
    out = []
    for rec in rows:
        info = rec.get("additional_info", {})
        in_audio = audio_for(rec.get("input_request", {}).get("audio_path"), is_tts=False)
        n_in += in_audio is not None

        pipes = {}
        for pk in ("pipeline1", "pipeline2"):
            tts_audio = audio_for((rec.get(pk, {}).get("tts") or {}).get("output"), is_tts=True)
            n_tts += tts_audio is not None
            pipes[pk] = build_pipeline(rec, pk, tts_audio)

        scores = [
            (pipes[pk] or {}).get("eval", {}).get("overall_score") if (pipes[pk] or {}).get("eval") else None
            for pk in ("pipeline1", "pipeline2")
        ]
        delta = abs(scores[0] - scores[1]) if None not in scores else None

        out.append({
            "annotation_id": rec["annotation_id"],
            "topic": info.get("topic"),
            "description": info.get("description"),
            "request_text": rec.get("input_request", {}).get("text"),
            "expected_voice_style": info.get("ideal_voice_response"),
            "audio": in_audio,
            "score_delta": delta,
            "pipeline1": pipes["pipeline1"],
            "pipeline2": pipes["pipeline2"],
        })
    print(f"{len(out)} records — {n_in} input clips, {n_tts} TTS clips")
    return out


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation viewer</title>
<style>
  :root {
    --bg:#f7f7f8; --card:#fff; --border:#e3e3e6; --fg:#1c1c1f; --muted:#6b6b72; --accent:#3b5bdb;
    --grey:#f0f0f2; --asr:#eef2f7; --llm:#f0f0f2; --tts:#f2eef7; --eval:#ecf6ee; --vote:#fdf8e6;
    --s1:#c92a2a; --s2:#e8590c; --s3:#f08c00; --s4:#5c940d; --s5:#2b8a3e;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg:#161618; --card:#202023; --border:#34343a; --fg:#e9e9ec; --muted:#9a9aa2; --accent:#748ffc;
      --grey:#27272c; --asr:#1e2530; --llm:#27272c; --tts:#272130; --eval:#1f2a23; --vote:#2c2820;
    }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:14px/1.5 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
  header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--border);
           padding:10px 16px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; }
  button, select { font:inherit; padding:6px 12px; border:1px solid var(--border);
           background:var(--bg); color:var(--fg); border-radius:6px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  input[type=number] { font:inherit; width:5em; padding:6px; border:1px solid var(--border);
           background:var(--bg); color:var(--fg); border-radius:6px; }
  .counter { color:var(--muted); }
  .spacer { flex:1; }
  main { max-width:1280px; margin:0 auto; padding:16px; }
  .request { background:var(--card); border:1px solid var(--border); border-radius:10px;
             padding:14px 16px; margin-bottom:16px; }
  .request h2 { margin:0 0 4px; font-size:12px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted); }
  .request .topic { font-size:12px; color:var(--muted); margin-bottom:6px; }
  .request p { margin:0 0 10px; font-size:15px; }
  .request .style { font-size:12.5px; color:var(--muted); margin:0 0 10px; }
  .request .style b { color:var(--fg); font-weight:600; }
  audio { width:100%; margin-top:4px; }
  .pipelines { display:grid; grid-template-columns:1fr 1fr; gap:16px; align-items:start; }
  @media (max-width:820px) { .pipelines { grid-template-columns:1fr; } }
  .pipeline { background:var(--card); border:1px solid var(--border); border-radius:10px; overflow:hidden; }
  .pipeline > h3 { margin:0; padding:10px 14px; border-bottom:1px solid var(--border); font-size:13px;
                   display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; }
  .stage { padding:12px 14px; border-bottom:1px solid var(--border); }
  .stage:last-child { border-bottom:0; }
  .stage .label { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--muted);
                  margin-bottom:5px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .stage.asr { background:var(--asr); } .stage.llm { background:var(--llm); }
  .stage.tts { background:var(--tts); } .stage.eval { background:var(--eval); } .stage.voting { background:var(--vote); }
  .sub { font-size:12.5px; color:var(--muted); margin:6px 0 2px; }
  .sub b { color:var(--fg); font-weight:600; }
  .kv { display:flex; flex-wrap:wrap; gap:5px 14px; margin:4px 0 8px; font-size:12.5px; }
  .kv span { color:var(--muted); } .kv b { color:var(--fg); font-weight:600; }
  .notes { margin:6px 0; }
  .notes .n-label { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .overall { font-weight:500; }
  .pert { font-weight:700; font-size:10.5px; letter-spacing:.03em; border-radius:5px; padding:1px 7px;
          background:var(--s2); color:#fff; text-transform:uppercase; }
  .pert.none { background:var(--border); color:var(--muted); }
  .badge { font-weight:700; border-radius:6px; padding:2px 9px; color:#fff; font-size:13px; }
  .b1{background:var(--s1)}.b2{background:var(--s2)}.b3{background:var(--s3)}.b4{background:var(--s4)}.b5{background:var(--s5)}
  .b0{background:var(--muted)}
  .missing { color:var(--muted); font-style:italic; }
  .vbar { display:flex; align-items:center; gap:6px; font-size:11px; margin:2px 0; }
  .vlab { width:1em; color:var(--muted); text-align:right; }
  .vtrack { flex:1; height:10px; background:var(--border); border-radius:5px; overflow:hidden; }
  .vfill { display:block; height:100%; }
  .vcnt { width:1.5em; color:var(--muted); }
  .delta { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <button id="prev">&larr; Prev</button>
  <button id="next">Next &rarr;</button>
  <label>id <input type="number" id="jump"></label>
  <label>filter
    <select id="filter">
      <option value="all">all</option>
      <option value="delta">score disagreement &ge; 2</option>
      <option value="lowagr">low vote agreement (&lt; 60%)</option>
      <option value="ttspert">TTS perturbation present</option>
    </select>
  </label>
  <span class="spacer"></span>
  <span class="counter" id="counter"></span>
</header>
<main id="view"></main>
<script>
const DATA = __DATA__;
const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;" }[c]));
const pct = x => (x === null || x === undefined) ? "?" : (100 * x).toFixed(0) + "%";
const scoreBadge = v => {
  const n = Number.isFinite(v) ? Math.max(1, Math.min(5, Math.round(v))) : 0;
  return `<span class="badge b${n}">${v ?? "?"}</span>`;
};
const pertTag = t => t ? `<span class="pert">${esc(t)}</span>` : `<span class="pert none">no perturbation</span>`;

let view = DATA.map((_, i) => i);
let pos = 0;
try { const s = +localStorage.getItem("eval_viewer_pos"); if (s >= 0) pos = s; } catch (e) {}

function applyFilter(kind) {
  view = DATA.map((r, i) => i).filter(i => {
    const r = DATA[i];
    if (kind === "delta") return r.score_delta !== null && r.score_delta >= 2;
    if (kind === "lowagr") return ["pipeline1", "pipeline2"].some(pk => {
      const v = r[pk] && r[pk].eval && r[pk].eval.voting;
      return v && v.overall_score && v.overall_score.modal_rate < 0.6;
    });
    if (kind === "ttspert") return ["pipeline1", "pipeline2"].some(pk => r[pk] && r[pk].tts_perturbation);
    return true;
  });
  if (!view.length) view = DATA.map((_, i) => i);
  pos = 0;
  render();
}

function evalBlock(ev) {
  if (!ev) return `<div class="missing">not evaluated</div>`;
  const asr = ev.asr_assessment || {}, llm = ev.llm_assessment || {},
        tts = ev.tts_assessment || {}, comp = ev.compounding || {};
  const b = x => (x === undefined || x === null) ? "?" : String(x);
  const note = (l, t) => t ? `<div class="notes"><div class="n-label">${l}</div>${esc(t)}</div>` : "";
  return `
    <div class="kv"><span>ASR severity</span><b>${b(asr.severity)}</b>
      <span>meaning kept</span><b>${b(asr.meaning_preserved)}</b></div>
    <div class="kv"><span>coherent</span><b>${b(llm.answer_coherent)}</b>
      <span>hallucinated</span><b>${b(llm.answer_hallucinated)}</b>
      <span>intent&harr;answer</span><b>${b(llm.intent_matches_answer)}</b>
      <span>serves need</span><b>${b(llm.answer_serves_true_need)}</b></div>
    <div class="kv"><span>spoken severity</span><b>${b(tts.spoken_severity)}</b>
      <span>meaning kept</span><b>${b(tts.spoken_meaning_preserved)}</b>
      <span>style matches</span><b>${b(tts.voice_style_matches_expected)}</b>
      <span>style apt</span><b>${b(tts.voice_style_appropriate)}</b>
      <span>delivery serves</span><b>${b(tts.delivery_serves_true_need)}</b></div>
    <div class="kv"><span>primary failure stage</span><b>${esc(comp.primary_failure_stage)}</b></div>
    ${note("ASR notes", asr.notes)}
    ${note("LLM notes", llm.notes)}
    ${note("TTS notes", tts.notes)}
    ${note("Compounding", comp.notes)}
    <div class="notes"><div class="n-label">Overall &nbsp; ${scoreBadge(ev.overall_score)}</div>
      <div class="overall">${esc(ev.overall_notes)}</div></div>`;
}

function votingBlock(v) {
  if (!v) return "";
  const os = v.overall_score || {};
  const dist = os.distribution || {};
  const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
  const bars = [1, 2, 3, 4, 5].map(s => {
    const c = dist[s] || 0;
    return `<div class="vbar"><span class="vlab">${s}</span>
      <span class="vtrack"><span class="vfill b${s}" style="width:${(100 * c / total).toFixed(0)}%"></span></span>
      <span class="vcnt">${c || ""}</span></div>`;
  }).join("");
  return `<div class="stage voting">
    <div class="label">Majority vote &nbsp; ${v.n_votes ?? "?"} runs${v.temperature != null ? ` @ T=${v.temperature}` : ""}</div>
    ${bars}
    <div class="kv">
      <span>chosen</span><b>${os.chosen ?? "?"}</b>
      <span>modal agr.</span><b>${pct(os.modal_rate)}</b>
      <span>pairwise agr.</span><b>${pct(os.pairwise_rate)}</b>
      <span>mean&plusmn;std</span><b>${os.mean ?? "?"} &plusmn; ${os.std ?? "?"}</b>
    </div></div>`;
}

function pipelineCol(n, p) {
  if (!p) return `<div class="pipeline"><h3>Pipeline ${n}</h3><div class="stage missing">not present</div></div>`;
  const ev = p.eval;
  const score = ev && !("_parse_error" in ev) ? scoreBadge(ev.overall_score) : `<span class="badge b0">?</span>`;
  return `
    <div class="pipeline">
      <h3><span>Pipeline ${n}</span> <span>${score}</span></h3>
      <div class="stage asr">
        <div class="label">ASR transcription ${pertTag(p.asr_perturbation)}</div>
        ${esc(p.asr_text) || '<span class="missing">&mdash;</span>'}
      </div>
      <div class="stage llm">
        <div class="label">LLM answer ${pertTag(p.llm_perturbation)}</div>
        ${esc(p.llm_answer) || '<span class="missing">&mdash;</span>'}
        ${p.llm_intent ? `<div class="sub">intent: <b>${esc(p.llm_intent)}</b></div>` : ""}
      </div>
      <div class="stage tts">
        <div class="label">TTS &mdash; spoken text ${pertTag(p.tts_perturbation)}</div>
        ${esc(p.spoken_text) || '<span class="missing">&mdash;</span>'}
        <div class="sub">delivered voice style: <b>${esc(p.delivered_voice_style)}</b></div>
        ${p.tts_impact ? `<div class="sub">perturbation impact: ${esc(p.tts_impact)}</div>` : ""}
        ${p.tts_audio ? `<audio controls preload="none" src="${p.tts_audio}"></audio>` : ""}
      </div>
      <div class="stage eval"><div class="label">Selected evaluation</div>${evalBlock(ev)}</div>
      ${ev ? votingBlock(ev.voting) : ""}
    </div>`;
}

function render() {
  if (!view.length) return;
  pos = (pos + view.length) % view.length;
  try { localStorage.setItem("eval_viewer_pos", pos); } catch (e) {}
  const r = DATA[view[pos]];
  document.getElementById("counter").textContent =
    `${pos + 1} / ${view.length}  ·  annotation_id ${r.annotation_id}` +
    (r.score_delta !== null ? `  ·  |Δscore| ${r.score_delta}` : "");
  document.getElementById("jump").value = r.annotation_id;
  document.getElementById("view").innerHTML = `
    <div class="request">
      <h2>Input request</h2>
      ${r.topic ? `<div class="topic">${esc(r.topic)}${r.description ? " — " + esc(r.description) : ""}</div>` : ""}
      <p>${esc(r.request_text)}</p>
      ${r.expected_voice_style ? `<div class="style">expected voice style: <b>${esc(r.expected_voice_style)}</b></div>` : ""}
      ${r.audio ? `<audio controls preload="none" src="${r.audio}"></audio>`
                : `<div class="missing">audio not embedded</div>`}
    </div>
    <div class="pipelines">
      ${pipelineCol(1, r.pipeline1)}
      ${pipelineCol(2, r.pipeline2)}
    </div>`;
  window.scrollTo(0, 0);
}

const byId = new Map(DATA.map((r, i) => [r.annotation_id, i]));
document.getElementById("prev").onclick = () => { pos--; render(); };
document.getElementById("next").onclick = () => { pos++; render(); };
document.getElementById("jump").onchange = e => {
  const i = byId.get(+e.target.value);
  if (i === undefined) return;
  const vp = view.indexOf(i);
  if (vp !== -1) { pos = vp; render(); }
};
document.getElementById("filter").onchange = e => applyFilter(e.target.value);
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") { pos--; render(); }
  if (e.key === "ArrowRight") { pos++; render(); }
});
render();
</script>
</body>
</html>
"""


def main(args):
    if not args.eval.is_file():
        raise SystemExit(
            f"{args.eval} not found — run the evaluator first:\n"
            f"    python src/llm_evaluation/single_pipeline.py"
        )
    records = build_records(args)
    if not records:
        raise SystemExit(f"no records in {args.eval}")
    html = HTML.replace("__DATA__", json.dumps(records, ensure_ascii=False))
    args.out.write_text(html, encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    if args.link_audio:
        print(f"wrote {args.out} ({mb:.1f} MB) — serve it:  "
              f"cd {args.out.parent} && python -m http.server")
    else:
        print(f"wrote {args.out} ({mb:.1f} MB) — open it in a browser")
        if mb > 50:
            print("  (large — re-run with --link-audio, or --limit N, to shrink it)")


if __name__ == "__main__":
    main(parse_args())
