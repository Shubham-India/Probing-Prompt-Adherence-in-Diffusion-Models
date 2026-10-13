# Attention Re-weighting Experiment

## 1. Maine yeh experiment kyun kiya?

Main experiment mein mujhe dikha ki Stable Diffusion 1.5 har baar prompt ki saari conditions follow nahi karta. Kabhi koi object image mein missing hota hai, aur kabhi colour galat object ko mil jaata hai.

Attention maps dekhne par mujhe laga ki `vase` ya `blue` jaise words ki attention aur image mein unke intended regions ke beech kuch relationship ho sakta hai. Isliye maine ek simple experiment try kiya: bina model ko train kiye, generation ke time object aur colour words ki attention badha di.

Mera main question tha ki kya is change se generated images prompt ko zyada accurately follow karengi.

## 2. Experiment se pehle meri hypothesis kya thi?

Maine hypothesis ko images generate karne se pehle `report/protocol.md` ke Addendum A mein likh diya tha. Isse result dekhne ke baad success ki definition change karne ka risk kam hota hai.

Meri hypothesis H4 thi:

- CFG 7.5 par object aur colour tokens ki attention 2 guna karne se overall success kam se kam `+0.05` improve honi chahiye.
- Improvement ka confidence interval zero se upar hona chahiye.
- CLIP score mein 0.5 se zyada decrease nahi hona chahiye.

Agar improvement zero ya negative hoti, ya image quality ko noticeable nuksan hota, toh main hypothesis ko supported nahi maanta.

## 3. Attention ko kaise change kiya?

Is experiment mein maine prompt ke object words aur colour words ko select kiya. Jaise `cube`, `sphere`, `chair`, `vase`, `red` aur `blue`. Plural forms ko bhi handle kiya, jaise `balls`.

Count words jaise `two`, relations jaise `left of`, aur material ya background words jaise `wooden` aur `table` ko boost nahi kiya.

### Code mein kya change hua?

Stable Diffusion ke UNet mein text cross-attention layers hoti hain. Maine har denoising step par conditional branch mein selected tokens ke pre-softmax attention scores mein `log(s)` add kiya.

Yahan `s` attention boost ka scale hai. Is change ke baad softmax dobara normalize hota hai, jisse selected tokens ko zyada attention mil sakti hai.

Unconditional branch ko change nahi kiya gaya.

Iske liye `src/intervene.py` mein `ReweightCrossAttention` aur `build_bias` implement kiye gaye. Attention bias ka shape `[batch, 1, 1, 77]` hai. Unconditional branch ke liye bias zero rehta hai, aur conditional branch mein selected token positions ko boost milta hai.

Maine teen scales test kiye:

- `s = 1`: control, jisme attention ka koi extra boost nahi hota.
- `s = 2`: main intervention condition.
- `s = 4`: stronger boost, taaki dekha ja sake ki zyada boost dene se kya hota hai.

Kaun se token positions boost hue, unka record `samples.jsonl` mein `boosted_token_positions` ke naam se save hota hai. Example ke liye, `a red cube and a blue sphere` mein positions `[2, 3, 6, 7]` ho sakti hain.

Is approach mein koi training, fine-tuning ya gradient-based optimization nahi hai. Yeh attention re-weighting ka ek gradient-free version hai. Attend-and-Excite jaise gradient-based approach ko maine is experiment mein test nahi kiya.

### Sabhi objects ko same boost kyun diya?

Pehle maine check kiya tha ki kya prompt mein baad mein mention kiya gaya object zyada frequently missing hota hai. Mere data ne is idea ko reliably support nahi kiya, kyunki mention order ka effect category aur object identity ke saath mixed tha.

Isliye maine kisi ek object ko target karne ke bajaye sabhi selected object aur colour tokens ko same rule ke under boost kiya.

## 4. Experiment ka design

Comparison fair rakhne ke liye maine main experiment ke same prompts, seeds aur generation settings use ki.

- 28 prompts aur har prompt ke 6 seeds, yaani seeds 0 se 5.
- CFG 7.5 aur 40 denoising steps.
- DDIM scheduler.
- Same seed ke liye same initial noise.
- Teen conditions: `rw1`, `rw2` aur `rw4`.

Har condition mein 168 images generate hui, isliye total 504 images thi.

Maine control condition ko main experiment ki normal pipeline se bhi compare kiya, jisme CFG 7.5 aur 40 steps wali images thi.

Analysis mein prompt-cluster bootstrap confidence intervals, prompt-level differences par Wilcoxon test, Cohen's \(d_z\) aur do intervention strengths ke liye Holm correction use ki gayi.

Primary comparison `rw2` versus `rw1` tha, aur main outcome overall prompt success tha. Presence, binding, counting, spatial aur CLIP score ko secondary outcomes ke roop mein dekha gaya.

## 5. Maine experiment kaise complete kiya?

Sabse pehle maine main results mein check kiya ki later-mentioned objects sach mein zyada missing hote hain ya nahi. Is pattern ka reliable evidence nahi mila, isliye maine sabhi selected objects aur colours par same boost rule rakha.

Uske baad `src/intervene.py` mein intervention code banaya. Ismein attention bias generate karne, images banane aur `samples.jsonl` mein settings save karne ka logic hai. Script mein resume support bhi hai.

### Pehle chhota test chalaya

Full run se pehle `outputs/intervention_smoke/` mein ek smoke test kiya. Ismein cube-sphere prompt, 6 seeds, control aur 4x boost test hue.

Control image normal pipeline se pixel-identical thi, jiska mean pixel difference 0.0 tha. Boosted image change hui; mean pixel difference 0–255 scale par lagbhag 12–14 tha. Token positions bhi check kiye aur woh sahi the.

Uske baad maine hypothesis, experiment design aur result ko reject karne ke criteria `report/protocol.md` ke Addendum A mein record kiye. Yeh full experiment ki images generate karne se pehle kiya gaya tha.

Full run mein 504 images generate hui, jisme ek image ko lagbhag 9 seconds lage. Run ke beech mein interruption aaya, isliye use resume karna pada. Is dauraan ek issue bhi hua, jiska detail agle section mein hai.

Images ko evaluate karne ke liye wahi OWL-ViT aur CLIP evaluator use kiya jo main experiment mein tha. Evaluation ke baad `src/analyze_intervention.py` se results nikale.

Ek important check yeh tha ki `rw1` control aur normal pipeline ki images compare ki jaayen. Sabhi 168 control images baseline se pixel-identical nikli. Iska matlab control mein attention boost na hone par naya code khud output ko change nahi kar raha tha.

Finally, intervention figure `main/figures/intervention.png`, report ka section 13 aur slide 9 update kiye gaye.

## 6. Run ke beech mein kya issue aaya?

Intervention run rukne ke baad resume hua. Galti se kuch waqt ke liye do generation processes ek hi output folder mein chalne lagi.

Isse kuch metadata lines duplicate ho gayi thi.

Maine duplicate lines remove ki aur har generated image verify ki. Sabhi images 512×512 resolution mein thi aur koi image corrupt nahi mili. Final folder mein 504 unique images hain, yaani har condition ki 168 images.

Generation deterministic thi, isliye duplicate writes se aayi images identical thi.

Is experience se ek baat clear hui: **ek output folder par ek hi generation process chalani chahiye.** Isse metadata conflicts aur duplicate records se bacha ja sakta hai.

## 7. Experiment ke results

### Primary success rate

Neeche har category aur condition ka mean success rate aur 95% confidence interval diya hai.

| Category | Control (s=1) | s=2 | s=4 |
|---|---|---|---|
| Presence | 0.62 [0.25, 0.96] | 0.79 [0.54, 1.00] | 0.75 [0.46, 1.00] |
| Binding | 0.11 [0.00, 0.25] | 0.14 [0.03, 0.28] | 0.14 [0.00, 0.31] |
| Counting | 0.44 [0.22, 0.67] | 0.47 [0.25, 0.70] | 0.47 [0.28, 0.67] |
| Spatial | 0.31 [0.11, 0.53] | 0.28 [0.14, 0.44] | 0.36 [0.17, 0.56] |
| Multi-constraint | 0.14 [0.00, 0.31] | 0.11 [0.00, 0.28] | 0.06 [0.00, 0.17] |
| **All prompts** | **0.30 [0.20, 0.42]** | **0.33 [0.21, 0.45]** | **0.33 [0.21, 0.45]** |

Overall results mein control ka mean success 0.30 tha, jabki `s=2` aur `s=4` dono mein 0.33 tha. Yaani observed improvement lagbhag 0.02 tha, exact reported contrast ke according.

### Control ke comparison mein result

Primary comparison mein:

- `s = 2`: difference `+0.02`, 95% CI `[-0.04, +0.09]`, Wilcoxon p = `0.378`, Holm-adjusted p = `0.757`.
- `s = 4`: difference `+0.02`, 95% CI `[-0.05, +0.10]`.

`s=2` condition mein 9 prompts ka success improve hua aur 5 prompts ka decrease hua. `s=4` mein 8 prompts improve hue aur 5 mein decrease hua.

Kisi bhi category mein Holm-adjusted p-value 0.05 se kam nahi tha.

### Kya meri main hypothesis support hui?

Nahi. H4 support nahi hui.

Maine experiment se pehle kam se kam `+0.05` improvement aur zero se upar confidence interval ka criterion rakha tha. Observed improvement sirf `+0.02` tha aur confidence interval zero ko include karta hai.

Binding aur multi-constraint prompts mein bhi expected overall improvement nahi dikhi. Isliye main yeh claim nahi kar sakta ki attention boost karne se overall prompt adherence improve ho gayi.

### Secondary results

`s=2` ko control se compare karne par:

- **Object presence:** `+0.09`, 95% CI `[+0.01, +0.17]`, Holm p = `0.021`.
- **Colour binding:** lagbhag `+0.02`.
- **Counting aur spatial:** koi statistically significant difference nahi mila; confidence intervals zero ko include karte hain.

Object presence ka result interesting hai, lekin ise confirmed finding nahi kehna chahiye. Yeh multiple secondary outcomes mein se ek tha aur in outcomes ke beech multiple-comparison correction nahi ki gayi thi. Isliye is result ko naye prompts ya independent confirmation set par test karna zaroori hai.

### CLIP score aur seed diversity

Image quality ke proxy ke roop mein CLIP score bhi compare kiya gaya.

- `s=2` par CLIP score ka difference `+0.07` tha, CI `[-0.28, +0.47]`.
- `s=4` par difference `-0.05` tha.

In results se CLIP score mein clear change establish nahi hua.

Seed diversity values ye thi:

- Control: `0.137`
- `s=2`: `0.142`
- `s=4`: `0.145`

Yeh differences chhote hain. Sirf in numbers se kisi strong diversity effect ka conclusion nahi nikalna chahiye.

### Images ko dekhne par kya dikha?

`main/figures/intervention.png` mein chaar prompts ke examples dikhaye gaye hain. Ye prompts predefined selection rule se choose kiye gaye the aur har example ke liye seed 0 use hua.

Chair aur vase wale examples mein attention boost karne se present object ka appearance change hua, lekin jo doosra object missing tha woh automatically generate nahi hua.

Yeh ek useful observation hai: selected word ki attention badha dena zaroori nahi ki model ko missing object sahi tarah generate karne ke liye enough information de.

## 8. `outputs/intervention/` folder mein kya hai?

Folder ke andar main files aur unka use:

```text
outputs/intervention/
├── intervention.md
├── images/
├── samples.jsonl
├── detections.jsonl
├── eval.jsonl
└── results/
    ├── per_image.csv
    ├── intervention_by_condition.csv
    ├── intervention_contrasts.csv
    ├── intervention_per_prompt.csv
    ├── intervention_diversity_per_prompt.csv
    └── intervention_numbers.json
```

- `images/`: 504 generated PNG images. Filenames ke aakhir mein `__rw1`, `__rw2` ya `__rw4` hota hai.
- `samples.jsonl`: generation settings ka record, including re-weighting scale aur boosted token positions.
- `detections.jsonl`: detector ke raw boxes, CLIP colour predictions, CLIP scores aur embeddings.
- `eval.jsonl`: har constraint ka pass/fail result.
- `per_image.csv`: har image ki ek row, aur condition ka naam.
- `intervention_by_condition.csv`: category aur condition ke hisaab se mean aur confidence interval.
- `intervention_contrasts.csv`: intervention aur control ke beech differences.
- `intervention_per_prompt.csv`: har prompt ke liye `rw2-rw1` aur `rw4-rw1` comparison.
- `intervention_diversity_per_prompt.csv`: har condition ke hisaab se diversity measurements.
- `intervention_numbers.json`: report mein use hone wale calculated results.

Is structure se raw outputs, evaluation results aur final statistical analysis ko alag-alag trace karna aasaan hota hai.

## 9. Experiment ki limitations

Is experiment mein kuch limitations hain:

- Sirf CFG 7.5 aur 40 denoising steps use kiye gaye.
- Sirf do non-control attention boost strengths, `s=2` aur `s=4`, test ki gayi.
- Colour-only boost aur har object ke liye alag adaptive boost test nahi kiya.
- Attend-and-Excite jaise gradient-based methods ke saath comparison nahi hua.
- Object presence ka observed improvement exploratory hai; iske liye independent confirmation set nahi tha.
- OWL-ViT/CLIP evaluator re-weighting ke baad badli hui images par biased ho sakta hai. Har condition ke liye alag manual validation nahi ki gayi.
- Benchmark mein sirf 28 prompts hain aur har category mein lagbhag 4–6 prompts hain. Isliye category-wise results ko cautiously interpret karna chahiye.

Future mein zyada prompts, independent manual evaluation aur different attention strategies ke saath experiment repeat kiya ja sakta hai.

## 10. Experiment dobara kaise run karein?

Virtual environment ke Python se ye commands order mein chalao:

```powershell
.\.venv\Scripts\python src\intervene.py --out outputs\intervention --scales 1 2 4

.\.venv\Scripts\python src\evaluate.py --run outputs\intervention

.\.venv\Scripts\python src\analyze_intervention.py --run outputs\intervention --baseline outputs\main

.\.venv\Scripts\python src\figures.py --run outputs\main --intervention outputs\intervention
```

Dhyaan rahe ki ek hi output folder par ek waqt mein ek hi generation process chalana hai.

## 11. Viva ke liye short explanation

“Maine Stable Diffusion 1.5 mein generation ke time object aur colour words ki attention ko 2x aur 4x boost karke test kiya, bina model ko train kiye. Same prompts, seeds aur settings par control se comparison kiya. Overall success sirf `+0.02` badhi aur confidence interval zero ko include karta hai, isliye meri main hypothesis support nahi hui. Object presence mein ek exploratory improvement dikha, lekin usse confirm karne ke liye independent testing chahiye.”

