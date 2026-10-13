# Live Demo Guide

Demo start karne se pehle project folder open karo aur check kar lo ki GPU free hai ya nahi.

```powershell
cd D:\prompt_Project
nvidia-smi
```

`nvidia-smi` se pata chalega ki GPU memory kitni use ho rahi hai. Stable Diffusion ko lagbhag 3.5 GB GPU memory chahiye, isliye agar koi doosra process GPU use kar raha hai toh use band karna better hoga.

Demo se pehle ye dono open rakhna:
- `outputs\main\figures`
- `outputs\main\results\tables.md`

Agar live generation mein koi problem aaye, toh saved results se bhi demo continue kar sakte ho.

---

## Step 1: Benchmark dikhao (30 seconds)

```powershell
Get-Content data\prompts.json -TotalCount 25
```

Is file mein har prompt ka `id`, `category`, `prompt` aur `constraints` diya gaya hai.

**Demo mein bolo:**

“Yeh hamara benchmark hai. Har prompt ke saath uski conditions bhi structured format mein store hoti hain. For example, kaun se objects image mein hone chahiye aur unka colour kya hona chahiye. Evaluator prompt ke sentence ko dobara parse karne ke bajaye in predefined constraints ko use karta hai.”

## Step 2: Images generate karo (lagbhag 1.5 minute)

```powershell
python src\generate.py --out outputs\demo --limit-prompts binding_cube_sphere_v1 --seeds 1
```

Agar `python` ka wrong version run ho raha ho, toh virtual environment ka Python use karo:

```powershell
.\.venv\Scripts\python src\generate.py --out outputs\demo --limit-prompts binding_cube_sphere_v1 --seeds 1
```

Generation complete hone ke baad images ka folder open karo:

```powershell
explorer outputs\demo\images
```

**Demo mein bolo:**

“Is example mein prompt hai ‘a red cube and a blue sphere’. Hum CFG 2, 7.5 aur 15 ko compare kar rahe hain, saath hi 20 aur 40 denoising steps bhi test kar rahe hain. Is tarah total chhah images generate hoti hain. In images ko compare karke hum dekh sakte hain ki model objects aur unke colours ko kitni accurately represent karta hai. Kuch outputs mein colour binding fail ho sakti hai, jaise dono objects ek hi colour ke ho jaana.”

Model load hone aur images generate hone mein thoda time lag sakta hai. Tab tak contact sheet dikha kar benchmark aur experiment ka idea explain kar sakte ho.

Ek extra example generate karne ke liye:

```powershell
python src\generate.py --out outputs\demo --limit-prompts presence_dog_bicycle --seeds 1
```

Ya ek single image generate karne ke liye:

```powershell
python -c "import sys; sys.path.insert(0,'src'); from generate import load_pipeline, generate_image; p=load_pipeline('stable-diffusion-v1-5/stable-diffusion-v1-5'); generate_image(p,'a boy and a green chair',7.5,20,0,512).save('outputs/demo_boy_chair.png')"
```

## Step 3: Reproducibility check karo (20 seconds)

```powershell
$n = "binding_cube_sphere_v1__cfg7.5__st40__seed0.png"
Get-FileHash outputs\demo\images\$n
Get-FileHash outputs\main\images\$n
```

Dono files ke hashes same hone chahiye.

**Demo mein bolo:**

“Yahan main abhi generate hui image ka hash main experiment mein saved image ke hash se compare kar raha hoon. Dono hashes same hain, matlab dono files byte-for-byte identical hain. Isse pata chalta hai ki same setup aur settings ke saath is output ko reproduce kiya ja sakta hai.”

Dhyaan rahe ki hash tabhi same aayega jab generation setup deterministic ho aur model, scheduler, settings aur environment compatible hon. Agar hashes match nahi karte, toh settings aur saved run record check karo.

## Step 4: Image ka record dikhao (20 seconds)

```powershell
Get-Content outputs\main\samples.jsonl -TotalCount 1
```

Ismein `name`, `prompt`, `cfg`, `steps`, `seed`, `scheduler` aur `model_id` jaise details milengi.

**Demo mein bolo:**

“Har generated image ke saath uski settings ka record save hota hai. Isse hum check kar sakte hain ki image kis prompt, CFG, denoising steps aur seed se generate hui thi. Ye records generation resume karne mein bhi help karte hain, kyunki system identify kar sakta hai ki kaun se samples pehle se generate ho chuke hain.”

## Step 5: Evaluator ki accuracy dikhao (10 seconds)

```powershell
python src\validate_metrics.py --run outputs\main
```

Yeh command binding, count, presence aur spatial categories ke validation results dikhayegi.

**Demo mein bolo:**

“Humne evaluator ko 40 manually labelled images par validate kiya. Labels ka initial draft AI ki help se banaya gaya tha, jise baad mein review karke kuch corrections kiye gaye. Is validation ki limitations bhi hain: sample size chhota hai aur spatial category mein positive examples bahut kam hain.”

## Step 6: Main results dikhao (lagbhag 1 minute)

```powershell
notepad outputs\main\results\tables.md
explorer outputs\main\figures
```

Ye figures dikhao:
- `adherence_by_cfg_steps.png`
- `contact_sheet.png`
- `attention_maps_by_cfg.png`

**Demo mein bolo:**

“CFG 2 se 7.5 par overall adherence mein observed increase 0.07 tha. Lekin confidence interval zero ko include karta hai, isliye hum confidently nahi keh sakte ki improvement statistically significant thi. Results mein binding aur multi-constraint prompts comparatively difficult rahe.”

## Step 7: Attention intervention ka result dikhao (lagbhag 1 minute)

```powershell
explorer outputs\main\figures
notepad outputs\intervention\intervention.md
```

Yahan `intervention.png` dikhao.

**Demo mein bolo:**

“Is experiment mein humne object aur colour tokens ki attention weights ko 2 guna kiya aur results ko control condition ke saath compare kiya. Overall success mein observed increase 0.02 tha, jo pehle se decide kiye gaye 0.05 threshold se kam hai. Isliye overall intervention hypothesis ko support nahi mila. Object presence mein ek exploratory improvement dikha, lekin us result ko confirm karne ke liye aur testing chahiye.”

## Step 8: Tests chalao (optional, 30 seconds)

```powershell
python -m pytest -q tests
```

Expected output is run mein `7 passed` hai.

**Demo mein bolo:**

“Tests benchmark structure, scoring aur attention hook ke behaviour ko check karte hain. Attention hook ke liye ek important check yeh hai ki hook add karne se generated image change na ho.”

---

## Agar demo ke dauraan koi problem aaye

| Problem | Kya karna hai |
|---|---|
| `CUDA out of memory` | Doosre GPU processes band karke dobara try karo. |
| Generation bahut slow hai | Thoda wait karo. Zarurat pade toh generation skip karke saved results dikhao. |
| Python ka wrong version run ho raha hai | `.\.venv\Scripts\python` use karo. |
| Hugging Face symlink warning | Agar sirf warning hai aur process sahi chal raha hai, toh aam taur par continue kar sakte ho. |
| Kuch bhi run nahi ho raha | `outputs\main\figures` aur `outputs\main\results\tables.md` se saved results dikhao. |

### Backup ke liye

Demo se pehle Step 3 ke hash output aur Step 5 ke validation results ka screenshot le kar rakh lo. Live commands fail hone par ye kaam aayega.

### Demo ke baad temporary files delete karni hon

Sirf tab run karna jab `outputs\demo` ki files ki zarurat dobara nahi padegi.

```powershell
Remove-Item -Recurse -Force outputs\demo
```

## Demo ke time yaad rakhna

1. **Limitations honestly explain karna.** Benchmark mein 28 prompts hain, labels review kiye gaye the aur attention analysis correlation dikhata hai, causation prove nahi karta.
2. **Negative result ko bhi clearly report karna.** Intervention ne overall hypothesis ko support nahi kiya. Yeh bhi ek valid experimental finding hai.
3. **AI use ke baare mein transparent rehna.** Project banane mein Claude/AI assistant ki help li gayi thi. Presentation se pehle event ke AI-use rules check kar lena.

---

# Naya prompt benchmark mein kaise add karein?

Naya prompt add karne ke do tareeke hain.

## Method A: `prompts.json` mein manually entry add karo

Agar sirf ek ya do prompts add karne hain, toh `data\prompts.json` mein existing entries ke format mein nayi entry likh sakte ho. Is tareeke mein `build_benchmark.py` chalana zaroori nahi hai.

Example:

```json
{
  "id": "binding_apple_bottle_v1",
  "category": "binding",
  "prompt": "a green apple and a red bottle on a wooden table",
  "constraints": [
    {"type": "presence", "object": "apple"},
    {"type": "presence", "object": "bottle"},
    {"type": "binding", "object": "apple", "color": "green"},
    {"type": "binding", "object": "bottle", "color": "red"}
  ],
  "pair_group": "binding_apple_bottle",
  "n_objects": 2,
  "n_constraints": 4
}
```

## Method B: `build_benchmark.py` mein template add karo

Agar benchmark ko systematically expand karna hai, toh `src\build_benchmark.py` mein naya prompt template add karo aur script se benchmark generate karo. Isse prompts ka format consistent rahega aur benchmark dobara banana aasaan hoga.

## Naya prompt add karte waqt kya check karna hai?

- `id` unique hona chahiye, kyunki ye image filenames mein bhi use hota hai.
- `category` supported categories mein se ek honi chahiye: `presence`, `binding`, `counting`, `spatial` ya `multi`.
- Constraint types supported hone chahiye: `presence`, `count`, `binding` ya `spatial`.
- `n_constraints` ki value constraints ki actual ginti ke barabar honi chahiye.
- Colour-binding ke liye supported colours mein se hi choose karo: `red`, `blue`, `yellow`, `green`, `orange`, `purple`, `white`, `black`, `brown`, `pink`.
- Aise objects use karo jinhe OWL-ViT detect kar sake. `apple`, `bottle` aur `dog` jaise common objects se shuru karna aasaan rahega.
- Object ka naam prompt mein usi form mein hona chahiye jise attention analysis search karta hai, jaise `ball` ya `balls`.
- Counting prompt mein count bhi specify karo, jaise `"count": 3`.

Prompt add karne ke baad tests run karke confirm karo ki benchmark ka format sahi hai.

## Naye prompts add karne ke baad workflow

Commands ko isi order mein run karo. Same output folder use karne se existing samples ko resume logic ke through skip karna aur missing samples generate karna possible hota hai.

| Step | Command | Purpose |
|---|---|---|
| 1 | `python -m pytest -q tests` | Benchmark aur code ke tests check karta hai. |
| 2 | `python src\generate.py --out outputs\main` | Missing prompts ke liye images generate karta hai. |
| 3 | `python src\evaluate.py --run outputs\main` | Generated images ke detections aur scores calculate karta hai. |
| 4 | `python src\attention.py --run outputs\main --steps 40` | 40-step images ke attention maps extract karta hai. |
| 5 | `python src\analyze.py --run outputs\main` | Main analysis sab available prompts par dobara chalata hai. |
| 6 | `python src\analyze_attention.py --run outputs\main` | Attention analysis update karta hai. |
| 7 | `python src\intervene.py --out outputs\intervention --scales 1 2 4` | Intervention experiment ke samples generate karta hai. |
| 8 | Intervention evaluation aur analysis scripts | Intervention scores aur summaries update karte hain. |
| 9 | `python src\figures.py ...` | Figures dobara generate karta hai; exact arguments script ke usage ke mutabik dena. |
| 10 | `python report\build_report.py` aur `python report\make_slides.py` | Report aur slides update karta hai. |

**Important:** Ek hi output folder par ek time mein ek hi generation process chalao. Multiple processes ek saath chalne se files aur metadata mein conflicts ho sakte hain.

Image count aur runtime is baat par depend karte hain ki kitne naye prompts hain aur kitne samples pehle se generate ho chuke hain. Isliye naye prompts add karne ke baad expected counts ko fixed guarantee mat samajhna.

## Naye prompts add karne ke baad kya change hoga?

- **Results:** Overall adherence, confidence intervals, p-values aur CFG/steps comparisons dobara calculate honge.
- **Report text:** Automatically generated numbers update ho sakte hain. Lekin hard-coded references, jaise “28 prompts”, category-wise counts, methodology diagram aur `data.md`, manually check karne honge.
- **Report build:** Naye results ke baad kuch assertions fail ho sakte hain. Aisa ho toh pehle data aur assertion check karo, phir conclusions ko naye results ke according update karo.
- **Tests:** Prompt IDs unique hone chahiye aur categories supported set ke andar honi chahiye.
- **Validation:** Existing 40-image validation subset purane prompts par based hai. Naye prompts add karne se validation set automatically update ya representative nahi banega. Zarurat ho toh naye prompts se additional labelled validation examples bhi include karo.

Report aur slides present karne se pehle ek baar verify kar lena ki tables, figures aur written conclusions sab latest results se match karte hain.