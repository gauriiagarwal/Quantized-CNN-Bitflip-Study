# Bit-Flip Sensitivity of Quantized CNNs, with a Low-Cost Hardware-Aware Protection

**What happens to a neural network when one bit in its memory flips? And what is the cheapest way to protect it?**

This repository is a simulation study that answers those questions for a small digit-recognition network (LeNet-5, 61,706 weights, MNIST), stored in three number formats: **FP32, INT8 and INT4**. It also includes a small parity checker written in **Verilog**, and an interactive **Streamlit website** where anyone can break the network themselves.

- **Author:** Gauri Agarwal, B.Tech (Electronics Engineering, VLSI Design & Technology), Banasthali Vidyapith
- **Live demo:** _add your Streamlit link here_
- **Type of work:** simulation only. No real chip was irradiated, attacked or tested.

---

## Table of contents

1. [The idea in two minutes](#1-the-idea-in-two-minutes)
2. [Background, explained simply](#2-background-explained-simply)
3. [What I did, step by step](#3-what-i-did-step-by-step)
4. [How the faults are injected](#4-how-the-faults-are-injected)
5. [The protections I compared](#5-the-protections-i-compared)
6. [Results](#6-results)
7. [The hardware part: parity checker in Verilog](#7-the-hardware-part-parity-checker-in-verilog)
8. [The website](#8-the-website)
9. [Repository contents](#9-repository-contents)
10. [How to run everything](#10-how-to-run-everything)
11. [Limitations (please read)](#11-limitations-please-read)
12. [Related work](#12-related-work)
13. [Future work](#13-future-work)
14. [Glossary](#14-glossary)
15. [A note on how this was made](#15-a-note-on-how-this-was-made)
16. [License](#16-license)

---

## 1. The idea in two minutes

A neural network is not magic. After training, everything it "knows" is a long list of numbers called **weights**. This network has 61,706 of them. The computer keeps each weight in memory as a row of **bits**, tiny switches that are either 0 or 1.

Memory is not perfect. Radiation, heat, ageing, or even a deliberate attack (for example "Rowhammer") can flip a stored bit from 0 to 1 or from 1 to 0. If that bit belongs to a weight, the network now remembers a wrong number.

The question is: **does it matter?** My simulation shows that the answer depends heavily on **which bit** flips and **which number format** is used:

- In **FP32** (32 bits per weight), flipping one special bit can turn a small weight into a gigantic one. The network then breaks completely.
- In **INT8** and **INT4** (8 or 4 bits per weight), the same kind of flip barely changes the answer, because these formats can only hold a small range of numbers.

I then tested cheap ways of protecting the weights, and checked how much hardware one of them (a parity checker) would cost.

Here is the most famous example of the problem, using real numbers:

```
Weight 0.3 stored as FP32:  0 01111101 00110011001100110011010
Flip bit 30 (an exponent bit):
                            0 11111101 00110011001100110011010   ->  1.02 x 10^38
```

One flipped bit turns `0.3` into about `10^38`. That one huge weight drowns out everything else in the network.

---

## 2. Background, explained simply

### Bits and bit positions
A **bit** is a switch: 0 (off) or 1 (on). A number is stored as a row of bits. **Bit position** says which switch in the row you mean. Position 0 is the smallest switch, and the highest position matters most.

### Number formats
| Format | Bits per weight | Layout | Key idea |
|---|---|---|---|
| FP32 | 32 | 1 sign + 8 exponent + 23 mantissa | Can store very large and very small numbers, so it is delicate |
| INT8 | 8 | sign + 7 value bits, plus one scale per layer | Only 256 possible values, so a flip cannot make a huge number |
| INT4 | 4 | sign + 3 value bits, plus one scale per layer | Only 16 possible values |

The **exponent bits** of FP32 decide how many zeros a number has, which is why flipping one of them is so dangerous.

### Quantization
**Quantization** means storing a number in a smaller, simpler format to save memory and power. Here I use **symmetric per-tensor quantization**: each layer gets one scale factor, and each weight becomes a small integer. A real value is roughly `integer x scale`.

Example in INT8: the stored number `5` is `00000101`. If the first (sign) bit flips, it becomes `10000101`, which a computer reads as `-123`. A small number became a large wrong one, but it is still limited to the range -128 to 127.

### Why hardware designers care
Phones, cars, satellites, medical devices and cameras run networks on small chips, mostly in INT8 or INT4. Knowing exactly which bits are dangerous tells a chip designer where protection is worth paying for.

---

## 3. What I did, step by step

1. **Trained a CNN.** LeNet-5 style: conv1 (1 to 6 channels, 5x5, padding 2), conv2 (6 to 16, 5x5), fc1 (400 to 120), fc2 (120 to 84), fc3 (84 to 10), with ReLU and max-pooling. Exactly **61,706 parameters**. Clean accuracy on the 10,000 MNIST test images: **98.00%**.
2. **Stored the weights in three formats** (FP32, INT8, INT4) and wrote my own code that converts every weight to its exact stored bits and back.
3. **Injected faults** in two ways (see section 4) and measured the accuracy afterwards.
4. **Repeated with many random seeds** so the results are not a lucky accident.
5. **Compared protections** (section 5) at different bit error rates.
6. **Wrote a parity checker in Verilog,** tested it exhaustively in Icarus Verilog, and estimated its size with Yosys.
7. **Built a website** that lets anyone repeat a small version of these experiments live.

I wrote my own bit-level code instead of using an existing fault-injection library because I needed to choose the **exact bit position in each format**.

---

## 4. How the faults are injected

All faults are applied to the **stored weights** (not to the computation, and not to the inputs).

### Targeted fault ("one bit position")
Pick a layer and a bit position. Flip that bit in a random **2% of the weights** of that layer. Measure how many accuracy points are lost. Repeating this for every layer and every bit position gives a **layer x bit heatmap**, which shows where the network is fragile.

### Random bit errors ("BER")
Every stored bit, in every layer, flips independently with probability `p`, the **bit error rate**. I used `p` from `1e-4` to `1e-2`. For the parity method, one extra parity bit per weight is stored and can also flip. This models soft errors, for example from radiation, where flips are random.

### Reading the results
Accuracy is measured on the MNIST test digits. Results are averaged over random seeds. A damage of "1 point" means accuracy fell by 1 percentage point, for example from 98% to 97%.

---

## 5. The protections I compared

| Name | How it works | Extra cost |
|---|---|---|
| **Parity-zero** | Store 1 extra bit per weight that makes the count of ON switches even. When read back, if the count is odd, a bit has flipped, so the weight is **set to 0**. | +1 bit per weight (3.1% for FP32, 12.5% for INT8, 25% for INT4) |
| **Weight nulling** | Remember the largest original weight of each layer. When reading, any weight larger than that (or not a number) cannot be right, so it is **set to 0**. | No extra storage |
| **Top-k bit protection** | Protect only the few most dangerous bit positions, found from the heatmap. Assumes a protected bit never fails. | Small, depends on k |
| **Mixed precision** | Keep sensitive layers in a bigger format and the rest in a smaller one. | Some extra memory |
| **Activation clipping** | Clamp the values flowing between layers so one huge value cannot spread. | No extra storage |

A key idea: **setting a damaged weight to 0 is a small, harmless mistake**, while leaving a huge wrong weight is a disaster. Both parity-zero and weight nulling use this idea.

Note: weight nulling and parity-based protection are **known techniques** in the literature, not inventions of this project (see section 12).

---

## 6. Results

All numbers come from my Colab notebook (MNIST test set of 10,000 images, LeNet-5 with 61,706 weights). The layer x bit sweep uses 10 random seeds with 95% bootstrap confidence intervals; the random-error and attacker experiments are described below. Small differences of about 1 point between runs are normal.

### 6.1 Clean accuracy
| Format | Accuracy |
|---|---|
| FP32 | 98.00% |
| INT8 | 97.98% |
| INT4 | 98.00% |

### 6.2 One flipped bit position in 2% of one layer (10 seeds)
**FP32, bit 30 (the top exponent bit), accuracy lost in points (95% confidence interval):**

| Layer | Points lost |
|---|---|
| conv1 | 54.8 (37.0 to 72.1) |
| conv2 | 80.8 (75.4 to 86.0) |
| fc1 | 58.8 (52.4 to 63.6) |
| fc2 | 80.0 (76.4 to 83.6) |
| fc3 | 79.7 (76.2 to 82.7) |

**Top bit (sign bit) in the integer formats:**

| Format | Range over the 5 layers | Worst layer |
|---|---|---|
| INT8 (bit 7) | 0.23 to 1.21 points | fc3 |
| INT4 (bit 3) | 0.50 to 1.63 points | fc3 |

Out of 220 (format, layer, bit) cells, only **11** are clearly harmful (the lower end of the confidence interval is above 1 point): 9 in FP32 and 2 in INT4, none in INT8.

### 6.3 Random bit errors (10 seeds), accuracy after the fault
**FP32**

| Bit error rate | No protection | Parity-zero | Weight nulling |
|---|---|---|---|
| 1e-4 | 20.06% | 97.93% | 97.99% |
| 1e-3 | 10.75% | 58.67% | 97.77% |
| 1e-2 | 9.80% | 9.80% | **95.51%** |

**INT8**

| Bit error rate | No protection | Parity-zero | Weight nulling |
|---|---|---|---|
| 1e-3 | 97.84% | 97.80% | 97.84% |
| 1e-2 | 93.77% | 95.69% | 94.10% |

**INT4**

| Bit error rate | No protection | Parity-zero | Weight nulling |
|---|---|---|---|
| 1e-3 | 97.64% | 97.96% | 97.71% |
| 1e-2 | 92.66% | 97.42% | 92.91% |

### 6.4 A smart attacker (greedy bit search)
A gradient-guided greedy search (in the spirit of the Bit-Flip Attack, using 500 training images) picks the most harmful bits, up to 30 flips:

- **FP32:** the very first flip (conv1, bit 30) drops accuracy from **98.00% to 12.80%**. Random flips need about 17 flips before the mean accuracy falls below 90%.
- **INT8:** 87.97% after 30 greedy flips. **INT4:** 17.65% after 30 greedy flips.
- **A range-aware attacker** who knows about weight nulling and only uses sign-bit flips drops FP32 below 90% after 3 flips (44.30% after 10), so weight nulling can be bypassed.

### 6.5 Other protections I tested
- **Top-k bit protection (M2, ideal assumption that protected bits never fail):** FP32 at 1e-2 goes from 9.80% to 42.79% / 88.19% / 92.42% for the top 1 / 2 / 3 bits, at +6.2% / +12.5% / +18.8% storage.
- **Mixed precision (M3):** keeping the 2 most sensitive layers in INT8 and the rest in INT4 gave 95.53% at 1e-2 with 31.6 KB, against 95.09% for all-INT8 (60 KB) and 91.58% for all-INT4 (30 KB).
- **Activation clipping (M5):** did not help much for stored-weight faults (FP32 at 1e-2 stayed near 10%) and changed nothing for INT8 and INT4.
- **Parity-zero plus weight nulling (FP32):** worse than weight nulling alone at 1e-2 (89.11% against 95.51%), and it costs 3.1% extra storage.

### 6.6 Faults in the scale factor
INT weights are stored with a float32 scale factor. Flipping the top exponent bit of a scale factor is **catastrophic**: about 88 points of accuracy lost with one scale per layer (per-tensor). One scale per channel limits the damage in the later layers (fc3 about 30 points) but not in conv1 and conv2. So INT8 and INT4 are robust to flips in the stored weight bits, but not to flips in their scale factors.

### 6.7 Confidence: wrong and sure
Accuracy alone hides a risk. At FP32 with bit error rate 1e-3, parity-zero reaches 50.84% accuracy, but 48% of all predictions are wrong while the model is at least 90% sure. Weight nulling at the same setting: 0.45%.

### 6.8 A second dataset (Fashion-MNIST)
The same CNN design on Fashion-MNIST (clean accuracy FP32 87.24%, INT8 87.12%, INT4 82.05%) shows the same pattern, but the integer formats lose more from one bit: the worst INT8 sign bit costs about 5.0 points and the worst INT4 top bit about 5.7 points (against about 1 to 1.6 on MNIST). FP32 bit 30 costs about 61 points. At error rate 1e-2, FP32: none 10.00%, weight nulling 78.09%.

### 6.9 What these results mean
- **Format matters a lot.** One bit can destroy an FP32 network. INT8 and INT4 degrade slowly on MNIST, but less safely on a harder dataset.
- **Weight nulling is excellent for FP32** (9.8% to about 95.5% at 1e-2) and needs about 0% extra storage, because a damaged FP32 weight is usually absurdly large and easy to spot.
- **Parity-zero suits the integer formats.** It can only detect an odd number of flips, and it can only set the weight to 0, not repair it. In FP32 it can leave the model wrong and very sure.
- **No protection is complete.** A smart attacker can get around weight nulling, and scale-factor flips are not covered by any of the weight protections tested here.

---

## 7. The hardware part: parity checker in Verilog

The parity checker is a very small circuit. Given the stored bits (the weight plus the parity bit), it checks whether a bit flipped and outputs either the weight or zero:

```verilog
module parity_zero #(parameter N = 8)(
    input  [N:0]   stored,   // N weight bits + 1 parity bit
    output [N-1:0] weight,
    output         err
);
    assign err    = ^stored;               // odd number of ones = a bit flipped
    assign weight = err ? {N{1'b0}} : stored[N-1:0];
endmodule
```

- `^stored` is the XOR of all bits. It is 1 when the number of ON bits is odd.
- If `err` is 1, the weight is replaced with zeros.
- The module was tested **exhaustively** in Icarus Verilog (every possible stored value).
- The testbench checks every weight value with no fault, every single flip (all detected and read as 0) and every double flip (not detected, as expected), for 4-bit and 8-bit words. All tests pass.
- Its size was **estimated with Yosys** and compared with a plain multiplier of the same width:

| Format | Gates | LUT4 | Share of one multiplier | Extra storage |
|---|---|---|---|---|
| INT4 | 8 | 6 | 11.6% | +25% |
| INT8 | 16 | 11 | 4.5% | +12.5% |
| FP32 | 64 | 43 | 1.0% | +3.1% |

These are rough relative estimates, not a finished chip, and I did not run this on an FPGA.

---

## 8. The website

The Streamlit site has 9 sections:

| # | Section | What you find |
|---|---|---|
| 1 | Know the project | The story at two levels: "like a kid" and "expert" (a full flowchart) |
| 2 | Tools used | What each tool did |
| 3 | Basic terms | 25 flashcards, one line each |
| 4 | Applications | Where one flipped bit could matter |
| 5 | Try yourself | Type a number, upload any picture, pick a format, a fault and a protection, and watch the result |
| 6 | Future scope | What I would do next |
| 7 | Limitations | What this study does not show |
| 8 | Why I chose this project | The story behind the topic |
| 9 | GitHub | This repository |

**Try yourself** runs the same fault code as the experiments, live, on the saved model. You can type any number, upload a handwritten digit or any picture, then choose the number format, the layer, the bit position or the bit error rate, and the protection. It shows what the network says before the fault, after the fault, and with protection, plus a bit-level view of one damaged weight, a layer x bit heatmap, and an accuracy vs error-rate curve.

If you upload a flower or a cat, the network will still answer with a digit, because it only knows 0 to 9. What matters there is how the faults change the answer and how sure the network is.

---

## 9. Repository contents

| Path | What it is |
|---|---|
| `app.py` | Home page and page router of the website |
| `know.py` | Content for "Know the project", Applications, Future scope, Tools, Why, GitHub |
| `flashcards.py` | The 25 term cards and the limitations text |
| `style.py` | The dark dashboard design (CSS) |
| `tryit.py` | The "Try yourself" live demo |
| `faults.py` | The CNN, bit conversion, fault injection and protections (PyTorch, NumPy) |
| `cnn_mnist.pt` | Trained LeNet-5 weights used by the demo |
| `mnist_samples.npz` | 500 MNIST test digits used by the demo |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Dark theme settings |
| `notebooks/` | Colab notebooks for training and the experiments |
| `hardware/` | `parity_zero.v` and its testbench |
| `results/` | Result tables (CSV) and graphs (PNG), including `paper_matrix.csv` |
| `docs/` | `Colab_Codes_and_Outputs.md` (all main Colab codes with outputs and explanations), plus .docx/.pdf versions |
| `resume/` | Resume (PDF) |
| `make_files.py` | Backup script to re-create the demo model and digits |

---

## 10. How to run everything

### Run the website
You need Python 3.10 or newer.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser. The first start can take a moment because PyTorch loads.

On Windows you can also double-click `run.bat`.

### Re-create the model files
`cnn_mnist.pt` and `mnist_samples.npz` are included. If you want to re-create them, run `make_files.py` (needs `torchvision` and an internet connection for the MNIST download), or run the export cell at the end of the Colab notebook.

### Run the experiments
Open the notebooks in `notebooks/` in Google Colab and run the cells from top to bottom. Run the training cell first, because the later cells use the trained `model`.

### Run the Verilog test
With Icarus Verilog installed:

```bash
iverilog -o sim hardware/parity_zero.v hardware/tb_parity_zero.v
vvp sim
```
(Adjust the file names to match the files in `verilog/`.)

---

## 11. Limitations (please read)

I want to be clear about what this study does **not** show.

- **One small model, mostly one dataset.** LeNet-5 on MNIST is easy. I added a Fashion-MNIST check, but bigger networks (ResNet, transformers) and harder data may behave very differently.
- **Simulation only.** No real radiation or real chip was used. Real faults can flip several nearby bits at once, but I mostly flip bits independently.
- **Mostly stored weights.** The main experiments fault the stored weights. I tested scale-factor faults separately (section 6.6) and they are catastrophic, but I did not fault activations or the computation itself.
- **Limited attackers.** The greedy attacker is simple, and a smarter one may do better. A range-aware attacker already bypasses weight nulling (section 6.4).
- **Ideal assumptions in some protections.** For top-k protection I assumed a protected bit never fails.
- **Parity limits.** Parity detects only an odd number of flips and cannot repair anything. In FP32 it can leave the model wrong and confident (section 6.7).
- **Hardware numbers are estimates.** Gate and LUT counts come from Yosys, and the checker was only tested in simulation.
- **Limited statistics.** Results use 5 to 10 random seeds, so very small differences (under about 1 point) cannot be told apart from noise.

---

## 12. Related work

This project builds on a body of earlier research. I do **not** claim to have invented the individual techniques. `paper_matrix.csv` lists the papers I compared against and how each relates to this work. Examples:

- **Ares** (Reagen et al., DAC 2018) and **PyTorchFI** (Mahmoud et al., DSN-W 2020): fault-injection frameworks for neural networks.
- **Terminal Brain Damage** (Hong et al., USENIX Security 2019) and **Bit-Flip Attack** (Rakin et al., ICCV 2019): show that a few flipped bits can destroy a network.
- **One Flip Away from Chaos** (Gongye and Fei, IEEE HOST 2024): single points of failure in quantized networks, including scale factors.
- **FT-ClipAct** (Hoang et al., DATE 2020) and **Ranger** (Chen et al., DSN 2021): protection by restricting activation ranges.
- **Value-aware Parity Insertion ECC** (Lee and Yang, DATE 2022): describes parity insertion and weight nulling, so **weight nulling and parity-zero are not new here.**
- **In-place Zero-space Memory Protection for CNNs** (NeurIPS 2019): protection that uses no extra storage.

What this project adds is a small, fully reproducible study in one place: a layer x bit sensitivity view across FP32, INT8 and INT4, a comparison of cheap protections, the measured hardware cost of a parity checker, and an interactive demo.

---

## 13. Future work

- Repeat the study on larger networks and harder datasets.
- Test on real hardware, for example an FPGA, and later with a radiation or fault-injection setup.
- Test smarter attackers and multi-bit faults.
- Try real error-correcting codes and choose protection layer by layer.
- Add protection for the scale factors, and try more formats (FP16, BF16, INT2).
- Train the network to tolerate flipped bits (fault-aware training).
- Turn the scripts into a small tool that tells a chip designer which layers and bits to protect.

---

## 14. Glossary

| Term | One-line meaning |
|---|---|
| Bit | A switch that is 0 or 1 |
| Weight | One remembered number inside the network |
| Bit position | Which switch in the row (0 is the smallest) |
| FP32 | 32-bit floating-point number |
| INT8 / INT4 | 8-bit or 4-bit integer formats |
| Exponent bit | The FP32 bits that decide how large a number is |
| Quantization | Storing numbers in a smaller, simpler format |
| CNN | A network that looks at pictures |
| MNIST | A standard set of handwritten digit images |
| Bit flip | A stored bit changes from 0 to 1 or 1 to 0 |
| Fault injection | Flipping bits on purpose to see what breaks |
| BER | Bit error rate: the share of stored bits that flip |
| Sensitivity | How much accuracy drops when something is hit |
| Heatmap | A coloured table, where darker red means more damage |
| Parity | One extra bit that makes the number of ON bits even, so a flip can be detected |
| Weight nulling | Setting an impossibly large weight to 0 |
| Verilog | A language that describes chip circuits |
| LUT4 | A small building block of an FPGA, used to measure circuit size |

---

## 15. A note on how this was made

The study design, the experiments, the analysis and the results are my own work. I used an AI assistant (Claude) to help write parts of the code, including the website, and I review it and can explain what it does.

---

## 16. License

MIT License. See the `LICENSE` file.

---

**Contact:** Gauri Agarwal · [LinkedIn](https://www.linkedin.com/in/gauri-agarwal-a7821a301) · [GitHub](https://github.com/gauriiagarwal)
