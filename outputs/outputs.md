# `outputs/` folder ki poori explanation

Is folder mein **saari generated cheezein** hain: images, metadata, scores, attention maps, statistics, figures aur logs. Input data (prompts, validation labels) `data/` mein hai. Yahan kuch bhi haath se nahi likha gaya, sab commands se bana hai.

```
outputs/
├── outputs.md                  <- yeh file
├── main/                       <- ASLI main experiment (viva mein dikhao)
├── intervention/               <- ASLI intervention experiment (viva mein dikhao)
├── smoke/                      <- shuru ka chhota test (ignore)
├── intervention_smoke/         <- intervention code ka chhota test (ignore)
├── e2e/                        <- aakhri end-to-end test (ignore, par log mat hatao)
├── e2e_intervention/           <- e2e test ka intervention hissa (ignore)
├── pilot_eval/                 <- khaali folder, kabhi use nahi hua (ignore / delete)
├── main.log, pilot.log         <- generation ke logs
├── attention.log               <- attention extraction ka log
├── intervention.log, intervention_resume.log   <- intervention ke logs
├── e2e.log                     <- end-to-end test ka log
├── test_results.txt            <- pytest ka result
├── validation_metrics.txt      <- evaluator ki accuracy (manual labels ke against)
└── pilot_detections_noembed.jsonl.bak   <- purana backup (ignore)
```

---

## 1. `main/` — main experiment (sabse zaroori)

Yeh **1008 images** ka poora experiment hai: 28 prompts × 3 CFG × 2 steps × 6 seeds. Pehle pilot (seeds 0-1, 336 images) isi folder mein bana, phir wahi run poori 6 seeds tak badha. Isliye pilot alag folder mein nahi hai, woh `main/` ka hissa hai.

```
main/
├── images/            1008 PNG images
├── samples.jsonl      har image ka generation-record
├── detections.jsonl   detector ke raw boxes + CLIP colour + CLIP score
├── eval.jsonl         har constraint ka pass/fail
├── attention/         504 attention maps (.npz)
├── results/           saari statistics tables
└── figures/           saare plots
```

### `images/` (441 MB, git mein nahi jaata)
- 1008 files, naam ka format `promptID__cfg<value>__st<steps>__seed<n>.png`, jaise `binding_chair_vase_v1__cfg15__st20__seed0.png`.
- Yahi naam har file mein key ki tarah use hota hai: `samples.jsonl`, `detections.jsonl`, `eval.jsonl`, `attention/*.npz` (`st40` wali) aur `per_image.csv`. Isse image aur uska data kabhi mix nahi hota.

### `samples.jsonl` — generation record (1008 lines)
Har line ek image: `name`, `prompt_id`, `category`, `prompt`, `cfg`, `steps`, `seed`, `size` (512), `scheduler` (DDIM eta=0), `model_id`, `negative_prompt` (khaali), `seconds`, `torch` (version). Isse koi bhi image seed se dobara ban sakti hai, aur generation resume bhi isi se hota hai.

### `detections.jsonl` — raw detector output (1008 lines)
Har line mein `name`, `detections`, `clip_score` aur `clip_image_embedding`.
- `detections`: har queried object ke liye top boxes, har box mein `score`, `box` (xyxy pixels), aur `colors` (CLIP ke 10 colours par probability).
- `clip_score`: image aur prompt ka CLIP similarity.
- `clip_image_embedding`: 512 numbers, seed-diversity nikalne ke liye.
- **Decision:** boxes ko raw store kiya, pass/fail nahi. Isse thresholds badalne par detector dobara nahi chalana padta (`evaluate.py --rescore`).

### `eval.jsonl` — scored result (1008 lines)
`samples.jsonl` ke saare fields, aur uske saath:
- `constraint_results`: har constraint ke liye 0/1, `prompts.json` ke order mein.
- `all_satisfied`: 1 agar saare constraints poore hue.
- `clip_score`.

### `attention/` (147 MB)
- Sirf **504 files**: woh images jinke `steps` = 40 hain (1008 ki aadhi).
- Naam image jaisa, extension `.npz`. Har file mein do arrays: `words` (prompt ke content words, jaise red, chair, blue, vase, wooden, table) aur `maps` (shape `[n_words, 40 steps, 32, 32]`, float16): har word ka har denoising step par attention map.
- **Decision:** 40-step wali images isliye chunin ki har image ko hook ke saath dobara generate karna padta hai (~9 sec har), aur 40 steps par stages (early/middle/late) zyada fine hain. Isliye attention analysis mein step-count ka effect nahi dekha ja sakta.
- Stop words (a, the, on...) store nahi hote.

### `results/` — statistics (sab `analyze.py` aur `analyze_attention.py` se)
| File | Kya hai |
|---|---|
| `tables.md` | Report ki main tables (markdown) |
| `per_image.csv` | Har image ek row: sab outcomes (`presence_ok`, `binding_ok`, `count_ok`, `spatial_ok`, `all_satisfied`, `primary`...) |
| `adherence_by_cell.csv` | Category × CFG × steps ka mean success + 95% CI |
| `constraint_fraction_by_cell.csv` | Wahi, par "kitne constraints poore hue" ka fraction |
| `constraint_type_by_cell.csv` | Constraint type (presence/binding/count/spatial) ke hisaab se accuracy |
| `cfg_contrasts.csv`, `steps_contrasts.csv` | Paired comparisons: difference, CI, Cohen's d_z, Wilcoxon p, Holm p. Yahan `diff` = lower level − higher level (a − b), isliye negative matlab higher level behtar |
| `interaction.json` | CFG × steps interaction ka GEE (Wald) test |
| `counts.json` | Images ki ginti, successes, failures, exclusions |
| `clip_diversity_per_prompt.csv`, `clip_diversity_summary.csv`, `clip_diversity_contrasts.csv` | CLIP score aur seed-diversity ka CFG/steps ke saath badalna |
| `attention_features.csv` | Har image ke attention summaries (strength, top-10% share, entropy, box ratio, bind margin) har stage ke liye |
| `attention_top10_by_step.csv` | Localisation vs denoising step, CFG ke hisaab se |
| `attention_cfg_effect.csv` | CFG ka attention summaries par effect |
| `attention_auc.csv` | Attention summary se success/failure alag karne ka AUC |
| `attention_classifier_auc.json` | Sirf-attention classifier ka AUC vs category+CFG baseline |

### `figures/` — plots
| File | Kya dikhata hai |
|---|---|
| `adherence_by_cfg_steps.png` | Category-wise success vs CFG, steps ke saath, CI ke saath |
| `constraint_type_accuracy.png` | Constraint types ki accuracy CFG ke saath |
| `contact_sheet.png` | Har category ka ek median-difficulty prompt, seed 0, saare 6 settings, hare/laal frame (success/failure) |
| `attention_maps_by_cfg.png` | Ek binding prompt ke word maps, early/middle/late, CFG 2/7.5/15 |
| `attention_success_vs_failure.png` | Wahi prompt, ek success aur ek failure ka attention |
| `attention_summary.png` | Localisation vs step + AUC plot |
| `methodology.png` | Pipeline ka flow diagram |
| `intervention.png` | Intervention vs control (bars + example images) |
| `slide_*.png` | Slides ke liye bade fonts wale versions (adherence, attention maps, attention summary, contact sheet) |

**Decision:** example images (contact sheet, attention figures) ek fixed rule se chune gaye, mann se nahi, taaki sirf achhe examples na dikhein. Rule `src/figures.py` ke docstring mein hai. Intervention ka figure bhi yahin (`main/figures/`) hai, `intervention/` mein nahi.

---

## 2. `intervention/` — attention re-weighting experiment

Object aur colour words ke attention ko 1×, 2× aur 4× karke dekha. Setting: CFG 7.5, 40 steps, 28 prompts × 6 seeds × 3 conditions = **504 images**.

```
intervention/
├── images/            504 PNG (naam ke aakhir mein __rw1, __rw2 ya __rw4)
├── samples.jsonl      main jaisa record + `reweight` aur `boosted_token_positions`
├── detections.jsonl   detector output
├── eval.jsonl         scored result
└── results/
```
- `rw1` = **control**: wahi code jisme factor 1 hai. Isne baseline images ko 168/168 pixel-identical reproduce kiya, isliye comparison code ka nahi, sirf factor ka hai.
- `rw2` = primary condition, `rw4` = dose-response check.
- `samples.jsonl` ke extra fields: `reweight` (1, 2 ya 4) aur `boosted_token_positions` (kaun se token boost hue, jaise `[2, 3, 6, 7]` = red cube blue sphere).

`results/`:
| File | Kya hai |
|---|---|
| `per_image.csv` | Har image ek row, `condition` column ke saath (baseline, rw1, rw2, rw4). Baseline rows `main/` ke CFG 7.5 / 40-step images se aati hain |
| `intervention_by_condition.csv` | Category × condition ka mean success + CI |
| `intervention_contrasts.csv` | Intervention − control ke paired comparisons har outcome ke liye |
| `intervention_per_prompt.csv` | Har prompt ka rw2 − rw1 aur rw4 − rw1 |
| `intervention_diversity_per_prompt.csv` | Condition ke hisaab se seed-diversity |
| `intervention_numbers.json` | Report mein jaane wale saare numbers |

**Decisions:** intervention ek alag folder mein rakha taaki main data se mix na ho. Hypothesis pehle `report/protocol.md` ke Addendum A mein likhi gayi, images banne se pehle (pre-registration). Result: primary outcome par H4 **supported nahi**; object presence mein exploratory sudhar.

**Ek operational ghatna:** intervention run ek baar ruka aur do generation processes thodi der ek saath chali. Dobara aayi metadata lines hata di (pehli rakhi), har image 512×512 par khulti hai yeh check kiya, aur ab bilkul 504 unique images hain (168 per condition).

---

## 3. Test aur debug folders (viva mein dikhane ki zarurat nahi)

| Folder | Kya hai |
|---|---|
| `smoke/` | Pehla 1-prompt test: 6 images, taaki generation pipeline check ho |
| `intervention_smoke/` | Intervention code ka chhota test: 12 images (control aur 4x), yeh dekhne ke liye ki control baseline se match karta hai |
| `e2e/` | End-to-end test: 5 prompts × 2 seeds, generate se attention aur figures tak. 60 images, 30 attention maps. Ismein `adherence_by_cell.csv`, `attention_*`, `cfg_contrasts.csv` jaise results bhi hain, jo **chhote data ke hain, research results nahi** |
| `e2e_intervention/` | E2E ka intervention hissa, 60 images |
| `pilot_eval/` | Khaali folder, shuru mein bana, kabhi kaam nahi aaya |

Inhe delete kar sakte ho, asli results par asar nahi padega. Bas `e2e.log` mat hatao.

---

## 4. Logs aur chhoti files

| File | Kya hai |
|---|---|
| `pilot.log` | Pilot generation (336 images) ka log |
| `main.log` | Baaki main generation ka log |
| `attention.log` | Attention extraction ka log, jisme bit-identical check hai (`max pixel diff 0`) |
| `intervention.log`, `intervention_resume.log` | Intervention generation ke logs |
| `e2e.log` | End-to-end test ka log, jiske aakhir mein `E2E-DONE` likha hai. README ka status isi se aata hai |
| `test_results.txt` | `pytest` ka output (7 passed). README aur defense notes ise padhte hain |
| `validation_metrics.txt` | `validate_metrics.py` ka output, final manual labels ke against |
| `pilot_detections_noembed.jsonl.bak` | Pilot ke detections ka purana backup, jab CLIP embeddings abhi store nahi hote the. Embeddings jodne ke baad detections dobara nikaale gaye. Ab zarurat nahi |

---

## Folder aur code ka rishta (workflow)

| Step | Command | Kya banta hai |
|---|---|---|
| Generate | `generate.py --out outputs\main` | `main/images`, `main/samples.jsonl` |
| Evaluate | `evaluate.py --run outputs\main` | `detections.jsonl`, `eval.jsonl` |
| Attention | `attention.py --run outputs\main --steps 40` | `main/attention/` |
| Statistics | `analyze.py`, `analyze_attention.py` | `main/results/` |
| Figures | `figures.py --run outputs\main --intervention outputs\intervention` | `main/figures/` |
| Intervention | `intervene.py --out outputs\intervention`, `evaluate.py`, `analyze_intervention.py` | `intervention/` |

## Kya git mein jaata hai aur kya nahi
- `.gitignore` mein `outputs/*/images/` hai, yaani saari images git mein nahi jaati (`main/images`, `intervention/images` waghera). Baaki outputs (jsonl, csv, figures, attention) git mein ja sakte hain.
- Images deterministic hain: `samples.jsonl` ke prompt, CFG, steps aur seed se dobara ban sakti hain.
- Viva ke liye laptop par hi dikhana, ya `outputs/main` aur `outputs/intervention` ko zip karke le jaana.

## Viva mein kya dikhana hai (kramwar)
1. `main/samples.jsonl` ki ek line aur `main/images` ki ek image: "yeh ek generated image aur uska record hai".
2. `main/eval.jsonl` ki ek line: "yeh uske constraints ka pass/fail hai".
3. `main/figures/contact_sheet.png` aur `attention_maps_by_cfg.png`: visual saboot.
4. `main/results/tables.md`: numbers aur confidence intervals.
5. `intervention/results/intervention_contrasts.csv` ya `figures/intervention.png`: negative result, jise dikhana imaandari dikhata hai.
6. `e2e.log`: "aakhir mein poori pipeline ka end-to-end test bhi chalaya."

## Ek note
`test_results.txt` aur `validation_metrics.txt` dono ko report ke rebuild ke time padha jaata hai. Agar kisi command ke beech mein ruk jaane se yeh khaali ho jaayein, to `pytest` aur `validate_metrics.py` dobara chalao, warna `build_report.py` fail hoga.
