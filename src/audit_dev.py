"""Blinded human audit of the first-pass evaluator (J1) on a generated run (E01), for the DEV audit that must precede the protocol freeze.
Differs from audit_sample.py/validate_human.py (kept for the pilot): the annotator answers a NEUTRAL question (a colour, a count, an object, yes/no)
instead of confirming a target, can say "object not visible", and the scorer re-weights the stratified sample to the population.
Commands
  sample  : python src/audit_dev.py sample --eval <eval.jsonl> --images <dir> --prompts <prompts.json> --out <dir> [--seed 1] [--quota presence=160 binding=160 count=0 spatial=0]
            (quota 0 = census of that type). Writes <out>/annotator_A/{label.html,images/}, <out>/annotator_B/ (double-annotation subset),
            <out>/key/{audit_key.csv,strata.json,manifest.json}. Annotator folders never contain predictions, targets, arms, CFG, seed or scene ids.
  pilot   : python src/audit_dev.py pilot --eval outputs/main/eval.jsonl --images outputs/main/images --prompts data/prompts.json --subset data/validation_subset.json --out <dir>
            second-review package for the pilot validation labels (all constraints of the 40 images; no practice items, no sampling).
  score   : python src/audit_dev.py score --key <out>/key/audit_key.csv --a labels_A.csv [--b labels_B.csv] --out report.json
  score-pilot : python src/audit_dev.py score-pilot --key <dir>/key/audit_key.csv --b labels_B.csv --existing data/validation_labels.json --subset data/validation_subset.json
No primary judge is chosen here: the report states, per constraint type, whether the predeclared criterion (accuracy >= 0.85) is met."""
import argparse
import csv
import hashlib
import html
import json
import math
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

PALETTE = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "other colour"]
THRESHOLD = 0.85          # predeclared in docs/AUDIT_PROTOCOL_DEV.md (RB 9.0 / 12.5)


# ---------------------------------------------------------------- questions and the mapping from answers to truth
def question(c, rng):
    t = c["type"]
    if t == "presence":
        return dict(qtype="yn", text=f"Is there at least one {c['object']} in the image?", options=[("yes", "yes"), ("no", "no"), ("amb", "ambiguous / cannot tell")])
    if t == "count":
        opts = [(str(i), str(i)) for i in range(10)] + [("10+", "10 or more"), ("amb", "cannot tell")]
        return dict(qtype="count", text=f"How many {c['object']}s are in the image? (count every distinct one)", options=opts)
    if t == "binding":
        opts = [(f"col:{p}", p) for p in PALETTE] + [("absent", f"no {c['object']} visible"), ("amb", "mixed / ambiguous")]
        return dict(qtype="colour", text=f"What colour is the {c['object']}? (if there are several, the colour of most of them)", options=opts)
    if t == "spatial":
        pair = [c["left"], c["right"]]
        rng.shuffle(pair)
        opts = [(f"obj:{pair[0]}", f"the {pair[0]}"), (f"obj:{pair[1]}", f"the {pair[1]}"), ("absent", "one or both not visible"), ("amb", "ambiguous / cannot tell")]
        return dict(qtype="left", text=f"Which one is further to the LEFT in the image? (compare the horizontal centres of the {pair[0]} and the {pair[1]})", options=opts)
    raise ValueError(t)


def truth(c, response):
    """(ok, present) from a neutral answer. ok = the constraint holds (joint with presence, as the legacy judge defines it); None = ambiguous."""
    r = (response or "").strip()
    if r in ("", "amb", "unclear"):
        return None, None
    t = c["type"]
    if t == "presence":
        return (1, 1) if r == "yes" else (0, 0)
    if t == "count":
        n = 10 if r == "10+" else int(r)
        return int(n == c["n"]), int(n > 0)
    if t == "binding":
        if r == "absent":
            return 0, 0
        return int(r == f"col:{c['color']}"), 1
    if t == "spatial":
        if r == "absent":
            return 0, 0
        return int(r == f"obj:{c['left']}"), 1
    raise ValueError(t)


# ---------------------------------------------------------------- sampling
def population(rows, prompts, subset=None):
    items = []
    for r in rows:
        if subset is not None and r["name"] not in subset:
            continue
        cons = prompts[r["prompt_id"]]["constraints"]
        for i, (c, pred) in enumerate(zip(cons, r["constraint_results"])):
            items.append(dict(name=r["name"], i=i, type=c["type"], pred=int(pred), unc=int(r.get("uncertain") or 0), c=c,
                              objpres=r.get("objects_present") if c["type"] in ("binding", "spatial") else None))
    return items


def stratum(it):
    return f"{it['type']}|pred{it['pred']}|unc{it['unc']}|pres{it['objpres'] if it['objpres'] is not None else '-'}"


def allocate(items, quotas, rng):
    """Equal allocation over the non-empty strata of each type (round-robin, capped by stratum size); quota 0 = census."""
    by_type = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_type[it["type"]][stratum(it)].append(it)
    chosen, strata = [], {}
    for t, st in by_type.items():
        for v in st.values():
            rng.shuffle(v)
        size = sum(len(v) for v in st.values())
        q = quotas.get(t, 0) or size
        q = min(q, size)
        taken = {k: 0 for k in st}
        keys = sorted(st)
        while sum(taken.values()) < q:
            for k in keys:
                if sum(taken.values()) >= q:
                    break
                if taken[k] < len(st[k]):
                    taken[k] += 1
        for k in keys:
            chosen += st[k][: taken[k]]
            strata[k] = dict(N=len(st[k]), n=taken[k])
    return chosen, strata


def opaque(seed, name):
    return hashlib.sha1(f"{seed}|{name}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------- annotator package
HTML = """<!doctype html><meta charset="utf-8"><title>Image audit</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:16px auto;padding:0 12px}img{max-width:100%;max-height:60vh;border:1px solid #888}
button{font-size:16px;margin:4px;padding:8px 12px;cursor:pointer}#q{font-size:20px;margin:10px 0}.k{opacity:.6;font-size:12px}#top{display:flex;justify-content:space-between}</style>
<div id="top"><span id="prog"></span><span><button onclick="back()">Back (b)</button><button onclick="dl()">Download labels CSV</button></span></div>
<div id="start"><p>Annotator ID: <input id="aid" size="14"> (e.g. your initials). Labels are NOT AI-assisted: do not use any AI tool for the answers.</p><button onclick="begin()">Start</button></div>
<div id="main" style="display:none"><div id="practice" class="k"></div><img id="im"><div id="q"></div><div id="opts"></div></div>
<script>
const ITEMS=__ITEMS__, PID="__PID__"; let ann="", i=0, R={}, t0=0;
const KEY="audit-"+PID; try{const s=JSON.parse(localStorage.getItem(KEY)||"null"); if(s){R=s.R;ann=s.ann;i=s.i;}}catch(e){}
if(ann){document.getElementById("aid").value=ann;}
function save(){try{localStorage.setItem(KEY,JSON.stringify({R,ann,i}));}catch(e){}}
function begin(){ann=document.getElementById("aid").value.trim(); if(!ann){alert("Enter an annotator ID");return;} document.getElementById("start").style.display="none"; document.getElementById("main").style.display="block"; show();}
function show(){ if(i>=ITEMS.length){document.getElementById("main").innerHTML="<h3>All items done. Click 'Download labels CSV' and keep the file.</h3>";dl();return;}
 const it=ITEMS[i]; document.getElementById("prog").textContent="item "+(i+1)+" / "+ITEMS.length+"  (answered "+Object.keys(R).length+")";
 document.getElementById("practice").textContent=it.practice?"PRACTICE ITEM (not analysed)":"";
 document.getElementById("im").src=it.image; document.getElementById("q").textContent=it.text; const o=document.getElementById("opts"); o.innerHTML="";
 it.options.forEach((p,k)=>{const b=document.createElement("button"); const h=k<9?String(k+1):(p[0]==="absent"?"n":(p[0]==="amb"||p[0]==="unclear")?"m":""); b.innerHTML=(h?"<span class='k'>"+h+"</span> ":"")+p[1]; b.onclick=()=>pick(p[0]); o.appendChild(b);}); t0=Date.now();}
function pick(v){const it=ITEMS[i]; R[it.id]={v:v,s:((Date.now()-t0)/1000).toFixed(1),t:new Date().toISOString()}; i++; save(); show();}
function back(){if(i>0){i--;save();show();}}
document.addEventListener("keydown",e=>{ if(document.getElementById("main").style.display==="none")return; if(e.key==="b"){back();return;} if(i>=ITEMS.length)return; const os=ITEMS[i].options.map(o=>o[0]); if(e.key==="n"&&os.includes("absent")){pick("absent");return;} if(e.key==="m"&&os.includes("amb")){pick("amb");return;} const k=parseInt(e.key); if(k>=1&&k<=9&&k<=os.length){pick(os[k-1]);}});
function dl(){let s="item_id,response,seconds,timestamp_utc,annotator\\n"; for(const id in R){s+=[id,R[id].v,R[id].s,R[id].t,ann].join(",")+"\\n";}
 const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([s],{type:"text/csv"})); a.download="labels_"+ann+"_"+PID+".csv"; a.click();}
</script>"""


def write_package(pdir, entries, images_dir, pid):
    pdir = Path(pdir)
    (pdir / "images").mkdir(parents=True, exist_ok=True)
    public = []
    for e in entries:
        dst = pdir / "images" / f"{e['image_id']}.png"
        if not dst.exists():
            shutil.copy(Path(images_dir) / f"{e['name']}.png", dst)
        public.append(dict(id=e["item_id"], image=f"images/{e['image_id']}.png", text=e["q"]["text"], options=e["q"]["options"], practice=int(e["practice"])))
    (pdir / "label.html").write_text(HTML.replace("__ITEMS__", json.dumps(public)).replace("__PID__", pid), encoding="utf-8")


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return ""


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def build(rows, prompts, images_dir, out, seed, quotas, subset=None, n_practice=20, n_double=100, args_for_manifest=None):
    out = Path(out)
    rng = random.Random(seed)
    pop = population(rows, prompts, subset)
    if subset is None:
        chosen, strata = allocate(pop, quotas, rng)
        taken = {(it["name"], it["i"]) for it in chosen}
        rest = [it for it in pop if (it["name"], it["i"]) not in taken]
        practice = rng.sample(rest, min(n_practice, len(rest)))
    else:
        chosen, practice = list(pop), []
        strata = {}
        for it in chosen:
            strata.setdefault(stratum(it), dict(N=0, n=0))
            strata[stratum(it)]["N"] += 1
            strata[stratum(it)]["n"] += 1
    entries = []
    ids = rng.sample(range(10**6), len(chosen) + len(practice))
    qrng = random.Random(seed + 7)
    for k, (it, pr) in enumerate([(x, 0) for x in chosen] + [(x, 1) for x in practice]):
        entries.append(dict(it, item_id=f"i{ids[k]:06d}", image_id=opaque(seed, it["name"]), q=question(it["c"], qrng), practice=pr))
    rng.shuffle(entries)
    entries.sort(key=lambda e: -e["practice"])      # stable: the practice round comes first, then the scored items in random order
    scored = [e for e in entries if not e["practice"]]
    by_type = defaultdict(list)
    for e in scored:
        by_type[e["type"]].append(e)
    double = set()
    if subset is None and n_double:
        per = max(1, n_double // max(len(by_type), 1))
        for t, es in by_type.items():
            double |= {e["item_id"] for e in rng.sample(es, min(per, len(es)))}
    key = out / "key"
    key.mkdir(parents=True, exist_ok=True)
    with open(key / "audit_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "name", "constraint_index", "type", "constraint_json", "judge_pred", "uncertain", "objects_present", "stratum", "N_h", "n_h", "practice", "double", "options_json"])
        for e in entries:
            s = stratum(e)
            w.writerow([e["item_id"], e["name"], e["i"], e["type"], json.dumps(e["c"]), e["pred"], e["unc"], e["objpres"], s,
                        strata.get(s, {}).get("N", ""), strata.get(s, {}).get("n", ""), e["practice"], int(e["item_id"] in double), json.dumps(e["q"]["options"])])
    (key / "strata.json").write_text(json.dumps(strata, indent=1))
    pid = hashlib.sha1(f"{seed}{len(entries)}{out}".encode()).hexdigest()[:8]
    write_package(out / "annotator_A", entries, images_dir, pid + "A")
    if double:
        write_package(out / "annotator_B", [e for e in entries if e["item_id"] in double or e["practice"]], images_dir, pid + "B")
    manifest = dict(seed=seed, quotas=quotas, n_items=len(scored), n_practice=len(practice), n_double=len(double), git_head=git_head(),
                    **(args_for_manifest or {}))
    (key / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------- scoring
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def read_key(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_labels(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["item_id"]] = r["response"].strip()
    return out


def kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def weighted_stats(recs, n_boot=2000, seed=0):
    """recs: dicts with stratum, N, n (per stratum), pred, ok. Returns population-weighted accuracy/TPR/TNR with a stratified bootstrap CI."""
    by = defaultdict(list)
    for r in recs:
        by[r["stratum"]].append(r)
    def est(sample):
        num = den = tp = fn = tn = fp = 0.0
        for s, rs in sample.items():
            if not rs:
                continue
            w = rs[0]["N"] / rs[0]["n"] if rs[0]["N"] and rs[0]["n"] else 1.0
            for r in rs:
                num += w * (r["pred"] == r["ok"]); den += w
                if r["ok"] == 1:
                    tp += w * (r["pred"] == 1); fn += w * (r["pred"] == 0)
                else:
                    tn += w * (r["pred"] == 0); fp += w * (r["pred"] == 1)
        return (num / den if den else float("nan"), tp / (tp + fn) if tp + fn else float("nan"), tn / (tn + fp) if tn + fp else float("nan"))
    point = est(by)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        draws.append(est({s: [rs[i] for i in rng.integers(0, len(rs), len(rs))] for s, rs in by.items()}))
    d = np.array(draws, float)
    ci = [[float(np.nanpercentile(d[:, j], 2.5)), float(np.nanpercentile(d[:, j], 97.5))] for j in range(3)]
    return dict(accuracy=point[0], accuracy_ci95=ci[0], sensitivity=point[1], sensitivity_ci95=ci[1], specificity=point[2], specificity_ci95=ci[2])


def score(key_rows, la, lb=None, n_boot=2000):
    rep, recs_by_type, pairs = {}, defaultdict(list), defaultdict(list)
    amb = defaultdict(int)
    tot = defaultdict(int)
    absent_judged_present = defaultdict(lambda: [0, 0])
    for r in key_rows:
        if r["practice"] == "1" or r["item_id"] not in la:
            continue
        c = json.loads(r["constraint_json"])
        a_ok, a_pres = truth(c, la[r["item_id"]])
        b_ok = None
        if lb is not None and r["item_id"] in lb:
            b_ok, _ = truth(c, lb[r["item_id"]])
            if a_ok is not None and b_ok is not None:
                pairs[r["type"]].append((a_ok, b_ok))
        tot[r["type"]] += 1
        if a_ok is None or (lb is not None and r["item_id"] in lb and (b_ok is None or b_ok != a_ok)):
            amb[r["type"]] += 1                       # ambiguous or annotator disagreement: excluded here, listed in disagreements
            continue
        recs_by_type[r["type"]].append(dict(stratum=r["stratum"], N=int(r["N_h"] or 0), n=int(r["n_h"] or 0), pred=int(r["judge_pred"]), ok=a_ok,
                                           present=a_pres, objpres=r["objects_present"], type=r["type"]))
    for t in ("presence", "count", "binding", "spatial"):
        rs = recs_by_type.get(t, [])
        k = sum(r["pred"] == r["ok"] for r in rs)
        row = dict(n_items_labelled=tot[t], n_used=len(rs), n_excluded_ambiguous_or_disagree=amb[t], unweighted_accuracy=(k / len(rs) if rs else None),
                   wilson95_unweighted=list(wilson(k, len(rs))), human_positive_share_in_sample=(sum(r["ok"] for r in rs) / len(rs) if rs else None))
        if rs:
            row.update(weighted_stats(rs, n_boot))
            row["criterion_met_accuracy_ge_0.85"] = bool(row["accuracy"] >= THRESHOLD)
        if t == "binding" and rs:
            cond = [r for r in rs if r["objpres"] == "1" and r["present"] == 1]          # judge says all objects present AND human sees this object
            row["conditional_binding_given_judge_all_present_and_human_present"] = dict(n=len(cond), accuracy=(sum(r["pred"] == r["ok"] for r in cond) / len(cond) if cond else None))
            fp = [r for r in rs if r["objpres"] == "1" and r["present"] == 0]
            row["judge_all_present_but_human_object_absent"] = dict(n=len(fp), of=len([r for r in rs if r["objpres"] == "1"]))
        if t in pairs and pairs[t]:
            row["double_annotation"] = dict(n_pairs=len(pairs[t]), raw_agreement=sum(x == y for x, y in pairs[t]) / len(pairs[t]),
                                            kappa=kappa([x for x, _ in pairs[t]], [y for _, y in pairs[t]]))
        rep[t] = row
    return rep


def score_pilot(key_rows, lb, existing, subset):
    rows = []
    for r in key_rows:
        if r["item_id"] not in lb:
            continue
        c = json.loads(r["constraint_json"])
        ok, _ = truth(c, lb[r["item_id"]])
        idx = subset.index(r["name"])
        old = existing["labels"][str(idx)][int(r["constraint_index"])]
        rows.append(dict(name=r["name"], i=int(r["constraint_index"]), type=r["type"], new=ok, old=old, judge=int(r["judge_pred"])))
    both = [r for r in rows if r["new"] is not None and r["old"] is not None]
    rep = dict(n_answered=len(rows), n_comparable=len(both), raw_agreement=(sum(r["new"] == r["old"] for r in both) / len(both) if both else None),
               kappa=kappa([r["old"] for r in both], [r["new"] for r in both]),
               second_reviewer_ambiguous=sum(r["new"] is None for r in rows), existing_null=sum(r["old"] is None for r in rows),
               by_type={t: dict(n=len([r for r in both if r["type"] == t]), agreement=(lambda xs: sum(r["new"] == r["old"] for r in xs) / len(xs) if xs else None)([r for r in both if r["type"] == t]))
                        for t in ("presence", "count", "binding", "spatial")},
               disagreements=[r for r in both if r["new"] != r["old"]],
               judge_vs_second_reviewer_accuracy=(sum(r["judge"] == r["new"] for r in both) / len(both) if both else None))
    return rep


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "pilot"):
        p = sub.add_parser(name)
        p.add_argument("--eval", required=True); p.add_argument("--images", required=True); p.add_argument("--prompts", required=True); p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=1); p.add_argument("--subset", default=None)
        p.add_argument("--quota", nargs="*", default=["presence=160", "binding=160", "count=0", "spatial=0"])
    s = sub.add_parser("score"); s.add_argument("--key", required=True); s.add_argument("--a", required=True); s.add_argument("--b", default=None); s.add_argument("--out", default=None)
    sp = sub.add_parser("score-pilot"); sp.add_argument("--key", required=True); sp.add_argument("--b", required=True); sp.add_argument("--existing", required=True); sp.add_argument("--subset", required=True)
    a = ap.parse_args()
    if a.cmd in ("sample", "pilot"):
        rows = [json.loads(l) for l in Path(a.eval).read_text().splitlines() if l.strip()]
        prompts = {p["id"]: p for p in json.loads(Path(a.prompts).read_text())}
        quotas = {k: int(v) for k, v in (q.split("=") for q in a.quota)}
        subset = json.loads(Path(a.subset).read_text()) if a.cmd == "pilot" else None
        man = build(rows, prompts, a.images, a.out, a.seed, quotas, subset=set(subset) if subset else None, n_double=0 if subset else 100,
                    args_for_manifest=dict(eval_sha256=file_sha(a.eval), prompts_sha256=file_sha(a.prompts), mode=a.cmd))
        print(json.dumps(man, indent=1))
    elif a.cmd == "score":
        rep = score(read_key(a.key), read_labels(a.a), read_labels(a.b) if a.b else None)
        txt = json.dumps(rep, indent=1)
        print(txt)
        if a.out:
            Path(a.out).write_text(txt)
    else:
        print(json.dumps(score_pilot(read_key(a.key), read_labels(a.b), json.loads(Path(a.existing).read_text()), json.loads(Path(a.subset).read_text())), indent=1))


if __name__ == "__main__":
    main()
