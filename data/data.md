1008 images 5 categories se bani hain: 28 prompts × 3 CFG × 2 steps × 6 seeds. Har prompt ki 36 images bani hain (3 × 2 × 6), isliye category ka total = prompts × 36

Category	Prompts	    Images	    Kya test karti hai
Presence	4	         144	    Kya sab objects bane?
Binding	    6	         216	    Colour sahi object par laga?
Counting	6	         216	    Sahi ginti bani?
Spatial	    6	         216	    Left/right sahi hai?
Multi	    6	         216	    Kai constraints ek saath
Total	    28	        1008	


data/
├── data.md                     
├── prompts.json                 <- 28 prompts + unke constraints
├── validation_subset.json       <- 40 images ke naam 
├── validation_labels.json       <- labels of those 40 images
└── validation_subsets_images/   <- copy of those 40 images
```

Poora chain: `prompts.json` → generate hui images (`outputs/main`) → 40 random images (`validation_subset.json`) → unke manual labels (`validation_labels.json`) → evaluator ki accuracy (`src/validate_metrics.py`).

---

## 1. `prompts.json` — benchmark

**Kaun banata hai:** `src/build_benchmark.py`. Prompts haath se nahi chune gaye, templates se bane hain. Dobara banane ke liye:
```powershell
.\.venv\Scripts\python src\build_benchmark.py
```
**Kaun padhta hai:** `src/generate.py` (images banane ke liye), `src/evaluate.py` (scoring ke liye), `src/analyze*.py`, `src/intervene.py`, aur tests.

**Isme 28 entries (prompts) hain.** Har entry mein yeh fields hain:

| Field | Matlab | Example |
|---|---|---|
| `id` | Prompt ka unique naam. Image ke naam mein bhi yahi aata hai | `binding_chair_vase_v1` |
| `category` | presence, binding, counting, spatial ya multi | `binding` |
| `prompt` | Asli text jo model ko diya jaata hai | `a red chair and a blue vase on a wooden table` |
| `constraints` | Structured list: is prompt mein kya-kya check karna hai | neeche dekho |
| `pair_group` | Controlled variants ko ek group mein jodta hai (colour ya side swap wale jode) | `binding_chair_vase` |
| `n_objects` | Prompt mein kitne alag objects hain | `2` |
| `n_constraints` | Constraints ki ginti. **Yahi "complexity" ki definition hai** | `4` |
| `count` | Sirf count wale prompts mein: kitni cheezein chahiye | `3` |

### `constraints` ke 4 types
Evaluation kabhi prompt ka text parse nahi karta. Woh seedha yeh structure padhta hai.

| Type | Fields | Check kya hota hai |
|---|---|---|
| `presence` | `object` | Kya yeh object image mein hai? |
| `count` | `object`, `n` | Kya bilkul `n` instances hain? |
| `binding` | `object`, `color` | Kya is object ka colour yahi hai? |
| `spatial` | `left`, `right` | Kya `left` wala object `right` wale ke baayen hai? |

Example, `binding_chair_vase_v1`:
```json
"constraints": [
  {"type": "presence", "object": "chair"},
  {"type": "presence", "object": "vase"},
  {"type": "binding",  "object": "chair", "color": "red"},
  {"type": "binding",  "object": "vase",  "color": "blue"}
]
```
Isi order mein automatic scoring ka result aur `validation_labels.json` ke labels aate hain.

### 28 prompts ki list

| Category | Prompts | Kya hain | `n_constraints` |
|---|---|---|---|
| **presence** (4) | dog/bicycle, cat/umbrella, horse/bench, bird/clock | "a dog and a bicycle" jaise | 2 |
| **binding** (6) | cube/sphere, cup/book, chair/vase, har ek **do baar** (colours swap) | "a red cube and a blue sphere on a wooden table" aur "a blue cube and a red sphere ..." | 4 |
| **counting** (6) | yellow apples aur red balls, har ek ke liye 2, 3, 4 | "three yellow apples on a wooden table" | 2 |
| **spatial** (6) | cat/dog, cup/book, car/bicycle, har ek ke do order | "a cat to the left of a dog" aur "a dog to the left of a cat" | 3 |
| **multi** (6) | 4 prompts: colour + left/right (cup/book, chair/vase, dono orders). 2 prompts: count + colour + doosra object | "a red cup to the left of a blue book ...", "two red balls and a blue cube ..." | 4 ya 5 |

Constraint types ka total (28 prompts mein): presence 42, binding 30, spatial 10, count 8.



---

## 2. `validation_subset.json` — kaun si 40 images?

- Sirf **image-names ki list** hai (images nahi). List ka **index (0 se 39)** hi har image ki pehchaan hai.
- Naam ka format: `promptID__cfg__steps__seed`, jaise `presence_horse_bench__cfg2__st20__seed0`.
- Yeh 40 images **pilot images (seed 0 aur 1)** mein se random chuni gayi: seed 123, har category se 8.
- Score dekhne se pehle chuni gayi thi, isliye selection par bias nahi hai.
- Asli image file: `outputs/main/images/<naam>.png`.

## 3. `validation_labels.json` — sahi jawab (manual labels)

- `labels` ke andar key `"0"` se `"39"` tak hai. Yeh keys `validation_subset.json` ke index hain.
- Har key ki value 0, 1 ya `null` ki list hai, **har constraint ke liye ek**, `prompts.json` wale order mein.
  - `1` = constraint poora hua.
  - `0` = nahi hua.
  - `null` = image se saaf nahi, isliye compare se bahar.
- Example, index 3 (`binding_cube_sphere_v2`, "a blue cube and a red sphere"): `[0, 1, 0, 1]` matlab cube nahi bana, sphere bana, cube blue nahi, sphere red hai.
- Example, index 11: `[1, null, null]` matlab cat dikha, par dog aur left/right saaf nahi.

**Labels ki history (kramwar):**
1. Pehla draft Claude ne images dekhkar likha, evaluator ka koi score dekhne se pehle.
2. Doosra pass (Claude ka, **blind nahi**): index 2 aur 9 ka ek-ek constraint ambiguous (`null`) hua, aur index 29 aur 35 ke peele-hare apples ko "yellow nahi" (`0`) mana gaya.
3. Aapke badlav: index 26 aur 32 (`vase is red`, `null` se `0`) aur index 38 (dog presence aur left/right order, `0` se `null`).
4. Sab badlav ek hi final `labels` set mein merge hain. File mein ab sirf `labels` hai, koi extra key nahi, aur code (`src/validate_metrics.py`) isi ko padhta hai.

Labels independent nahi hain (pehla draft AI ka hai). Agar aapne images khud dekhkar verify kiye hain, to `report/validation.md` mein jo bracket wali line hai usme likho ki kitni images khud check ki aur kab. Viva mein yehi sach batana: "pehla draft AI ki madad se bana, phir maine khud verify aur sudhaare."

## 4. `validation_subsets_images/` — 40 images ki copy

- `validation_subset.json` ki saari 40 images yahan same naam se copy hain (originals `outputs/main/images/` mein rahte hain).
- Sirf **dekhne ke liye** hai. Code isse nahi padhta, wo hamesha `outputs/main/images/` se padhta hai.
- Verify karte waqt: list se index ka naam lo, yahan woh `.png` kholo, `prompts.json` mein constraints dekho, aur `validation_labels.json` ke label se milao.
- Folder alphabetical dikhega, list ke order mein nahi. Isliye index list se hi dekhna.

---

## Validation ka maksad aur result kaise nikalta hai

**Maksad:** automatic evaluator (OWL-ViT + CLIP) par bharosa kitna kar sakte hain, yeh naapna.

```powershell
.\.venv\Scripts\python src\validate_metrics.py --run outputs\main
```
Yeh har constraint ke manual label ko automatic result (`outputs/main/eval.jsonl`) se compare karta hai. `null` skip hota hai. Phir constraint type (presence, binding, count, spatial) ke hisaab se accuracy, precision aur recall deta hai, final labels ke hisaab se.

Report ke numbers ke liye: `report/validation.md` aur `report/report.md` ka section 8.

Limitations:
- Ek annotator, 40 images.
- Spatial mein sirf 2 positive labelled constraints, isliye spatial ki accuracy kamzor tareeke se verify hui hai.
- Size attributes evaluate hi nahi hue.
- Yellow-green apples ko CLIP "green" ya "orange" bata deta hai, isliye counting prompts ka primary outcome sirf count constraint hai.
