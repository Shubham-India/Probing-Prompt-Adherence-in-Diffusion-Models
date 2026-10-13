# Where Does the Prompt Go? Probing Prompt Adherence in Diffusion Models

> Note (carried over from the pre-merge README, kept verbatim): 1008 images 5 categories se bani hain: 28 prompts × 3 CFG × 2 steps × 6 seeds. Har prompt ki 36 images bani hain (3 × 2 × 6), isliye category ka total = prompts × 36

This project looks at a simple question: **when a text prompt contains several requirements, how reliably does Stable Diffusion 1.5 follow them?** I tested how classifier-free guidance (CFG) scale and the number of denoising steps affect prompt adherence, including object presence, attribute binding, counting, spatial relationships, and prompts with multiple constraints. I also examined cross-attention maps and tested whether re-weighting attention for object and colour tokens could improve the results.

The experiment plan, hypotheses, proposed mechanism, and analysis choices are documented in [`report/protocol.md`](report/protocol.md). The findings and interpretation are in [`report/report.md`](report/report.md), and the evaluator checks are in [`report/validation.md`](report/validation.md). Presentation files and defence notes are in [`report/presentation.pptx`](report/presentation.pptx), [`report/speaker_notes.md`](report/speaker_notes.md), and [`report/defense.md`](report/defense.md).

> **Main finding:** Increasing CFG from 2 to 7.5 raised image–text similarity by 1.88 and reduced diversity from 0.21 to 0.14. Measured adherence increased by 0.07, but its 95% confidence interval (−0.01 to +0.15) includes zero, so the result is inconclusive. More denoising steps and CFG 15 did not provide a clear benefit. Attribute-binding and multi-constraint prompts remained difficult across settings. The attention re-weighting intervention changed overall success by only +0.02 (95% CI: −0.04 to +0.09), so **the pre-registered H4 hypothesis was not supported**. An improvement in object presence appeared in exploratory analysis, but it needs a separate confirmation test.

## Project structure

```text
configs/
  experiment.json             CFG scales, step counts, seeds and other settings

data/
  prompts.json                28 prompts with structured constraints
  validation_*.json           Manual validation labels and second-pass overrides

src/
  build_benchmark.py          Builds the prompt benchmark
  generate.py                 Generates images for each prompt/configuration/seed
  evaluate.py                 Runs OWL-ViT and CLIP scoring
  validate_metrics.py         Compares automated scores with manual labels
  attention.py                Extracts cross-attention maps
  intervene.py                Runs the attention re-weighting experiment
  analyze.py                  Calculates the main statistics and tables
  analyze_attention.py        Summarises attention patterns and AUCs
  analyze_intervention.py     Compares intervention results with the baseline
  report_numbers.py           Collects reported numbers from result files
  figures.py                  Generates figures from saved outputs

tests/                        Unit tests and a GPU attention test
report/
  protocol.md                 Hypotheses and analysis plan
  validation.md               Evaluator validation
  report.md                   Generated results and discussion
  defense.md                  Generated defence notes
  make_slides.py              Builds presentation slides
outputs/                      Generated images, metadata, evaluations and results
```

`outputs/` contains generated artifacts and is not versioned. The benchmark is built by `src/build_benchmark.py` from templates; it can be regenerated when needed.

## Requirements

The project was tested on **Windows 11**, **Python 3.13**, and an **NVIDIA GPU with at least 6 GB of VRAM**. The test machine was an RTX 3050 Laptop GPU with 6 GB. Allow roughly 10 GB of disk space for model files and outputs, and around 3.5 GB of free GPU memory for the tested setup.

The models are downloaded from Hugging Face on first use and stored locally under `models/`:

- `stable-diffusion-v1-5/stable-diffusion-v1-5` — fp16 model, approximately 2.1 GB. The model is distributed under the CreativeML OpenRAIL-M licence and is a mirror of the deprecated `runwayml` repository.
- `google/owlvit-base-patch32` — object detection.
- `openai/clip-vit-base-patch32` — image–text scoring.

Python 3.14 was not used because compatible PyTorch wheels were unavailable in this setup. The exact dependency versions used for the experiment are recorded in `requirements.txt`.

## Installation

Run these commands in PowerShell from the project folder. The `py` launcher must have Python 3.13 installed; otherwise, replace `py -3.13` with the path to your Python 3.13 executable.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.\.venv\Scripts\python -m pip install -r requirements.txt
```

If you need to reproduce the original package-by-package setup instead of using `requirements.txt`, the main libraries are `diffusers`, `transformers`, `accelerate`, `safetensors`, `scipy`, `pandas`, `matplotlib`, `statsmodels`, `scikit-learn`, `pytest`, and `python-pptx`.

## Running the experiment

Run commands from the project root. The generation and attention scripts can resume work from files already on disk. **Run only one generation process per output directory** to avoid conflicts.

### 1. Run the tests and build the benchmark

```powershell
.\.venv\Scripts\python -m pytest -q tests
.\.venv\Scripts\python src\build_benchmark.py
```

The recorded test run reported 7 passed tests and 1 warning in 28.78 seconds. The benchmark builder regenerates `data/prompts.json`.

### 2. Generate images

```powershell
# Small smoke test: 6 images
.\.venv\Scripts\python src\generate.py --out outputs\smoke --limit-prompts binding_cube_sphere_v1 --seeds 1

# Pilot run: 336 images, about 40 minutes on the tested setup
.\.venv\Scripts\python src\generate.py --out outputs\main --seeds 2

# Main run: resumes existing work and completes the 1,008-image grid
.\.venv\Scripts\python src\generate.py --out outputs\main
```

The full grid uses 28 prompts, 3 CFG settings, 2 step settings, and 6 seeds: `28 × 3 × 2 × 6 = 1,008` generated samples.

### 3. Evaluate and validate the generated images

```powershell
.\.venv\Scripts\python src\evaluate.py --run outputs\main
.\.venv\Scripts\python src\validate_metrics.py --run outputs\main
```

The evaluator writes detection and scoring files. Use `--rescore` with `src/evaluate.py` when only rescoring is needed.

### 4. Extract and analyse cross-attention

```powershell
.\.venv\Scripts\python src\attention.py --run outputs\main --steps 40 --check 5
.\.venv\Scripts\python src\analyze.py --run outputs\main
.\.venv\Scripts\python src\analyze_attention.py --run outputs\main
```

The attention run covers 504 samples at step 40. The `--check` option runs the bit-identity check used to confirm that enabling extraction does not change the generated image.

### 5. Run the attention intervention

```powershell
.\.venv\Scripts\python src\intervene.py --out outputs\intervention --scales 1 2 4
.\.venv\Scripts\python src\evaluate.py --run outputs\intervention
.\.venv\Scripts\python src\analyze_intervention.py --run outputs\intervention --baseline outputs\main
```

This compares attention re-weighting scales 1, 2, and 4 with the matched baseline. The recorded intervention run covered 504 images and took about 80 minutes on the tested setup.

### 6. Regenerate the report and slides

```powershell
.\.venv\Scripts\python src\figures.py --run outputs\main --intervention outputs\intervention
.\.venv\Scripts\python report\build_report.py
.\.venv\Scripts\python report\make_slides.py
```

The report builder creates `report/report.md`, `report/defense.md`, and `README.md`. The slide builder creates `report/presentation.pptx` and `report/speaker_notes.md`. Reported numerical values are collected from saved result files by `src/report_numbers.py`.

## What has been checked

The table below summarises the checks recorded in the project files. “Verified” means the stated artifact or check is present; it does not mean that the overall scientific hypothesis has been proven.

| Area | Status | Evidence / note |
|---|---|---|
| Hypothesis and analysis plan written before the main run; intervention plan recorded before its run | Verified | `report/protocol.md`, including the revision log and Addendum A |
| Benchmark covers five prompt categories with controlled variants | Verified | `data/prompts.json` contains 28 prompts; `tests/test_logic.py::test_benchmark_is_consistent` |
| Generation parameters and seeds are recorded; generation can resume | Verified | `outputs/main/samples.jsonl` contains 1,008 records; same-seed regeneration was reported as bit-identical |
| CFG and denoising steps tested with multiple seeds | Verified | `results/adherence_by_cell.csv` records the 3 × 2 × 6 × 28 design |
| Automated metrics compared with manual labels | Partly complete | `report/validation.md`; one annotator, 40 images, only two positive spatial-relation examples, and size attributes not evaluated |
| Cross-attention extraction checks token alignment and image identity | Verified in the recorded checks | `tests/test_attention_gpu.py`, `outputs/attention.log`, and 504 saved attention files |
| Attention patterns compared with adherence | Exploratory | `results/attention_*.csv` and `figures/attention_*.png` |
| Statistics reproduced from saved results | Verified | `src/analyze.py` creates CSV tables and `tables.md`; report generation checks the wording against the data |
| Figures generated from experiment outputs | Verified | `outputs/main/figures/` |
| Main hypothesis conclusively tested | Inconclusive | Only 4–6 prompts per category; confidence intervals include zero |
| Matched intervention and baseline comparison run | Completed; H4 not supported | `src/intervene.py`, `outputs/intervention/results/`, report section 13 |
| Overall adherence improved by attention re-weighting | Not demonstrated | Primary contrast: +0.02, 95% CI −0.04 to +0.09; object-presence result is exploratory |
| References checked | Checked | Report section 17 records a citation check dated 2026-10-10 |
| Report, slides, speaker notes, and defence material prepared | Verified in project files | `report/report.md`, `report/presentation.pptx`, `report/defense.md` |
| End-to-end smoke run (generate, attention, evaluate, analyse, intervene; 60 + 30 images) | Completed per log | `outputs/e2e.log` ends with `E2E-DONE`; not re-run after the latest code changes; not confirmatory evidence |
| Independent review of manual validation labels | Not started | A second reviewer still needs to check the labels |

## Limitations and next steps

The benchmark is useful for a controlled pilot, but it is still small. Each category has only 4–6 prompts, so the outcome can depend heavily on the chosen objects and wording. Some conditions have low success rates, which creates a floor effect and makes improvements hard to measure. Metric validation is also limited: the manual review used one annotator and 40 images, spatial relations had only two positive examples, and size attributes were not evaluated.

The experiments cover one diffusion model, scheduler, and resolution, with a relatively coarse CFG grid. The attention analysis is correlational, so it cannot show that an attention pattern caused a prompt to succeed or fail. The intervention was tested at one CFG setting, one step count, and one token-selection rule. The object-presence improvement was exploratory and has not been checked on a separate confirmation set.

The next useful steps are to obtain an independent review of the validation labels, expand the prompt set, and test the exploratory object-presence finding on new prompts and seeds. For now, the main conclusion should remain **inconclusive** rather than a claim that higher CFG or attention re-weighting solves compositional prompt adherence.
