"""Build a self-contained HTML viewer for the evaluator's predictions.

Reads the base-model and tuned-model inference jsonl (from `infer.py`) and emits
one HTML file: the shared input-request audio on top, then two columns
(pipeline1 / pipeline2), each cascading ASR -> LLM -> base-model evaluation ->
tuned-model evaluation -> gold score. Navigate with the arrows, the id box, or
the left/right keyboard keys. Audio is embedded as base64 so the file works by
double-clicking -- no server.

    python results/view_annotations.py
    python results/view_annotations.py --limit 40 --gold data/eval_data.jsonl
"""

import argparse
import base64
import json
import mimetypes
from pathlib import Path

HERE = Path(__file__).parent
SFT = HERE.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=HERE / "base_model" / "inference.jsonl")
    p.add_argument("--tuned", type=Path, default=HERE / "tuned_model" / "inference.jsonl")
    p.add_argument("--out", type=Path, default=HERE / "viewer.html")
    p.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="jsonl with pipeline*.overall_evaluation to show alongside the model's "
        "(defaults to data/eval_data.jsonl if it exists)",
    )
    p.add_argument("--audio-dir", type=Path, default=None, help="override dir for audio basenames")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-audio", action="store_true", help="skip audio entirely (smallest file)")
    p.add_argument(
        "--link-audio",
        action="store_true",
        help="symlink wavs into <out>_audio/ and reference them relatively instead of "
        "embedding (keeps the html small; needs `python -m http.server` to view)",
    )
    return p.parse_args()


def audio_data_uri(raw_path: str, audio_dir: Path | None) -> str | None:
    path = Path(raw_path)
    if audio_dir is not None:
        path = audio_dir / path.name
    if not path.is_file():
        return None
    mime = mimetypes.guess_type(path.name)[0] or "audio/wav"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def load_gold(path: Path | None) -> dict:
    if path is None or not path.is_file():
        return {}
    gold = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        gold[rec["annotation_id"]] = {
            pk: rec.get(pk, {}).get("overall_evaluation") for pk in ("pipeline1", "pipeline2")
        }
    return gold


def resolve_audio(raw_path: str, audio_dir: Path | None) -> Path:
    path = Path(raw_path)
    return audio_dir / path.name if audio_dir is not None else path


def link_audio(src: Path, link_dir: Path) -> str | None:
    if not src.is_file():
        return None
    link_dir.mkdir(parents=True, exist_ok=True)
    dst = link_dir / src.name
    if not dst.exists():
        dst.symlink_to(src.resolve())
    return f"{link_dir.name}/{src.name}"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def merge_pipeline(base_rec: dict, tuned_rec: dict, pk: str) -> dict | None:
    bp, tp = base_rec.get(pk), tuned_rec.get(pk)
    if not bp and not tp:
        return None
    ref = bp or tp
    return {
        "asr": ref.get("asr"),
        "llm": ref.get("llm"),
        "base_eval": (bp or {}).get("model_evaluation"),
        "tuned_eval": (tp or {}).get("model_evaluation"),
    }


def build_records(args) -> list[dict]:
    base_rows = _read_jsonl(args.base)
    tuned_by_id = {r["annotation_id"]: r for r in _read_jsonl(args.tuned)}
    if args.limit is not None:
        base_rows = base_rows[: args.limit]

    gold_path = args.gold or (SFT / "data" / "eval_data.jsonl")
    gold = load_gold(gold_path if gold_path.is_file() else None)
    link_dir = args.out.with_name(args.out.stem + "_audio")

    n_audio = 0
    out = []
    for rec in base_rows:
        ann_id = rec["annotation_id"]
        trec = tuned_by_id.get(ann_id, {})
        src = resolve_audio(rec["input_request"]["audio_path"], args.audio_dir)
        if args.no_audio:
            audio = None
        elif args.link_audio:
            audio = link_audio(src, link_dir)
        else:
            audio = audio_data_uri(str(src), None)
        n_audio += audio is not None
        out.append(
            {
                "annotation_id": ann_id,
                "request_text": rec["input_request"]["text"],
                "audio": audio,
                "gold": gold.get(ann_id),
                "pipeline1": merge_pipeline(rec, trec, "pipeline1"),
                "pipeline2": merge_pipeline(rec, trec, "pipeline2"),
            }
        )
    print(f"{len(out)} annotations, {n_audio} with audio")
    return out


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Annotation viewer</title>
<style>
  :root {
    --bg: #f7f7f8; --card: #fff; --border: #e3e3e6; --fg: #1c1c1f; --muted: #6b6b72;
    --accent: #3b5bdb;
    --grey: #f0f0f2; --eval-base: #fdf8e6; --eval-tuned: #ecf6ee; --gold: #eaf1fc;
    --s1:#c92a2a; --s2:#e8590c; --s3:#f08c00; --s4:#5c940d; --s5:#2b8a3e;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg:#161618; --card:#202023; --border:#34343a; --fg:#e9e9ec; --muted:#9a9aa2;
      --accent:#748ffc;
      --grey:#27272c; --eval-base:#2c2820; --eval-tuned:#1f2a23; --gold:#20252f;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  header { position: sticky; top: 0; z-index: 5; background: var(--card);
           border-bottom: 1px solid var(--border); padding: 10px 16px;
           display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  button { font: inherit; padding: 6px 12px; border: 1px solid var(--border);
           background: var(--bg); color: var(--fg); border-radius: 6px; cursor: pointer; }
  button:hover { border-color: var(--accent); }
  input[type=number] { font: inherit; width: 5em; padding: 6px; border: 1px solid var(--border);
                       background: var(--bg); color: var(--fg); border-radius: 6px; }
  .counter { color: var(--muted); }
  main { max-width: 1200px; margin: 0 auto; padding: 16px; }
  .request { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
             padding: 14px 16px; margin-bottom: 16px; }
  .request h2 { margin: 0 0 8px; font-size: 13px; text-transform: uppercase;
                letter-spacing: .04em; color: var(--muted); }
  .request p { margin: 0 0 10px; font-size: 15px; }
  audio { width: 100%; }
  .pipelines { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 780px) { .pipelines { grid-template-columns: 1fr; } }
  .pipeline { background: var(--card); border: 1px solid var(--border);
              border-radius: 10px; overflow: hidden; }
  .pipeline > h3 { margin: 0; padding: 10px 14px; border-bottom: 1px solid var(--border);
                   font-size: 13px; display: flex; justify-content: space-between;
                   align-items: center; gap: 10px; flex-wrap: wrap; }
  .pipeline > h3 .scores { display: flex; gap: 12px; font-size: 12px; font-weight: 600; }
  .pipeline > h3 .sc { display: flex; align-items: center; gap: 5px; color: var(--muted); }
  .stage { padding: 12px 14px; border-bottom: 1px solid var(--border); }
  .stage:last-child { border-bottom: 0; }
  .stage .label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em;
                  color: var(--muted); margin-bottom: 5px; }
  .stage.asr, .stage.llm { background: var(--grey); }
  .stage.base { background: var(--eval-base); }
  .stage.tuned { background: var(--eval-tuned); }
  .stage.gold { background: var(--gold); }
  .kv { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 4px 0 8px; font-size: 12.5px; }
  .kv span { color: var(--muted); }
  .kv b { color: var(--fg); font-weight: 600; }
  .notes { margin: 6px 0; }
  .notes .n-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .overall { font-weight: 500; }
  .badge { font-weight: 700; border-radius: 6px; padding: 2px 9px; color: #fff; font-size: 13px; }
  .b1{background:var(--s1)}.b2{background:var(--s2)}.b3{background:var(--s3)}.b4{background:var(--s4)}.b5{background:var(--s5)}
  .b0{background:var(--muted)}
  .missing { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<header>
  <button id="prev">&larr; Prev</button>
  <button id="next">Next &rarr;</button>
  <label>id <input type="number" id="jump"></label>
  <span class="counter" id="counter"></span>
</header>
<main id="view"></main>
<script>
const DATA = __DATA__;
const byId = new Map(DATA.map((r, i) => [r.annotation_id, i]));
let idx = 0;
try { const s = +localStorage.getItem("viewer_idx"); if (s >= 0 && s < DATA.length) idx = s; } catch (e) {}

const esc = s => String(s ?? "").replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
const scoreBadge = (v, label) => {
  const n = Number.isFinite(v) ? Math.max(0, Math.min(5, Math.round(v))) : 0;
  return `<span class="badge b${n}">${label ?? (v ?? "?")}</span>`;
};

function evalBlock(ev) {
  if (!ev) return `<div class="missing">no model evaluation</div>`;
  if (ev._parse_error) return `<div class="missing">unparseable model output</div>`;
  const obj = v => (v && typeof v === "object") ? v : (typeof v === "string" ? { notes: v } : {});
  const asr = obj(ev.asr_assessment), llm = obj(ev.llm_assessment), comp = obj(ev.compounding);
  const note = (label, txt) => txt ? `<div class="notes"><div class="n-label">${label}</div>${esc(txt)}</div>` : "";
  return `
    <div class="kv">
      <span>ASR severity</span> <b>${esc(asr.severity)}</b>
      <span>meaning kept</span> <b>${asr.meaning_preserved === undefined ? "?" : asr.meaning_preserved}</b>
      <span>primary failure</span> <b>${esc(comp.primary_failure_stage)}</b>
    </div>
    <div class="kv">
      <span>coherent</span> <b>${llm.answer_coherent ?? "?"}</b>
      <span>serves need</span> <b>${llm.answer_serves_true_need ?? "?"}</b>
      ${"answer_hallucinated" in llm ? `<span>hallucinated</span> <b>${llm.answer_hallucinated}</b>` : ""}
      ${"behaved_as_intended" in llm ? `<span>as intended</span> <b>${llm.behaved_as_intended}</b>` : ""}
    </div>
    ${note("ASR notes", asr.notes)}
    ${note("LLM notes", llm.notes)}
    ${note("Compounding", comp.notes)}
    <div class="notes"><div class="n-label">Overall notes &nbsp; ${scoreBadge(ev.overall_score)}</div>
      <div class="overall">${esc(ev.overall_notes)}</div></div>`;
}

function goldBlock(gold) {
  if (!gold) return "";
  const score = gold.overall_score;
  return `
    <div class="stage gold">
      <div class="label">GOLD Evaluation &nbsp; ${scoreBadge(score, String(score ?? "?").toUpperCase())}</div>
      <div class="overall">${esc(gold.overall_notes)}</div>
    </div>`;
}

function pipelineCol(n, stage, gold) {
  if (!stage) return `<div class="pipeline"><h3>Pipeline ${n}</h3>
    <div class="stage missing">not present</div></div>`;
  const badge = ev => ev && !ev._parse_error ? scoreBadge(ev.overall_score) : `<span class="badge b0">?</span>`;
  const goldScore = gold ? scoreBadge(gold.overall_score, String(gold.overall_score ?? "?").toUpperCase()) : "";
  return `
    <div class="pipeline">
      <h3><span>Pipeline ${n}</span>
        <span class="scores">
          <span class="sc">Base Model: ${badge(stage.base_eval)}</span>
          <span class="sc">Tuned Model: ${badge(stage.tuned_eval)}</span>
          ${gold ? `<span class="sc">Gold: ${goldScore}</span>` : ""}
        </span>
      </h3>
      <div class="stage asr"><div class="label">ASR transcription</div>${esc(stage.asr) || '<span class="missing">—</span>'}</div>
      <div class="stage llm"><div class="label">LLM answer</div>${esc(stage.llm) || '<span class="missing">—</span>'}</div>
      <div class="stage base"><div class="label">Base Model Evaluation</div>${evalBlock(stage.base_eval)}</div>
      <div class="stage tuned"><div class="label">Tuned Model Evaluation</div>${evalBlock(stage.tuned_eval)}</div>
      ${goldBlock(gold)}
    </div>`;
}

function render() {
  idx = (idx + DATA.length) % DATA.length;
  try { localStorage.setItem("viewer_idx", idx); } catch (e) {}
  const r = DATA[idx];
  document.getElementById("counter").textContent =
    `${idx + 1} / ${DATA.length}  ·  annotation_id ${r.annotation_id}`;
  document.getElementById("jump").value = r.annotation_id;
  const g = r.gold || {};
  document.getElementById("view").innerHTML = `
    <div class="request">
      <h2>Input request</h2>
      <p>${esc(r.request_text)}</p>
      ${r.audio ? `<audio controls preload="none" src="${r.audio}"></audio>`
                : `<div class="missing">audio not embedded</div>`}
    </div>
    <div class="pipelines">
      ${pipelineCol(1, r.pipeline1, g.pipeline1)}
      ${pipelineCol(2, r.pipeline2, g.pipeline2)}
    </div>`;
  window.scrollTo(0, 0);
}

document.getElementById("prev").onclick = () => { idx--; render(); };
document.getElementById("next").onclick = () => { idx++; render(); };
document.getElementById("jump").onchange = e => {
  const i = byId.get(+e.target.value);
  if (i !== undefined) { idx = i; render(); }
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft") { idx--; render(); }
  if (e.key === "ArrowRight") { idx++; render(); }
});
render();
</script>
</body>
</html>
"""


def main(args):
    records = build_records(args)
    if not records:
        raise SystemExit(f"no records in {args.base}")
    html = HTML.replace("__DATA__", json.dumps(records, ensure_ascii=False))
    args.out.write_text(html, encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    if args.link_audio:
        print(f"wrote {args.out} ({mb:.1f} MB) — serve it:  "
              f"cd {args.out.parent} && python -m http.server")
    else:
        print(f"wrote {args.out} ({mb:.1f} MB) — open it in a browser")


if __name__ == "__main__":
    main(parse_args())
