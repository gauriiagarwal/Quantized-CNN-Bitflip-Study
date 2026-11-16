"""Try yourself: dashboard page. Live fault injection on the saved CNN and saved MNIST digits.
Everything is simulation. Needs cnn_mnist.pt and mnist_samples.npz next to app.py."""
import os
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
RED, TEAL, AMBER, MUTED, PANEL, TEXT = "#FF5D5D", "#3DD6C6", "#FFD166", "#8B98A8", "#151C24", "#E6EDF3"
YOSYS = {"FP32": (64, 43, "1.0%"), "INT8": (16, 11, "4.5%"), "INT4": (8, 6, "11.6%")}  # gates, LUT4, % of a multiplier
PARITY_V = """module parity_zero #(parameter N = 8)(
    input  [N:0]   stored,   // N weight bits + 1 parity bit
    output [N-1:0] weight,
    output         err
);
    assign err    = ^stored;               // odd number of ones = a bit flipped
    assign weight = err ? {N{1'b0}} : stored[N-1:0];
endmodule"""


def _find(name):
    for d in (HERE, os.path.join(HERE, "results")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


@st.cache_resource(show_spinner="Loading model and digits...")
def get_assets():
    import torch
    import faults
    mp, sp = _find("cnn_mnist.pt"), _find("mnist_samples.npz")
    if not mp or not sp:
        return {"error": "missing", "need": [n for n, p in (("cnn_mnist.pt", mp), ("mnist_samples.npz", sp)) if not p]}
    try:
        model = faults.load_model(mp)
    except Exception as e:  # shown to the user, not hidden
        return {"error": "model", "msg": str(e)}
    z = np.load(sp)
    imgs = lab = None
    for k in z.files:
        a = z[k]
        if imgs is None and a.ndim >= 3 and a.shape[-1] == 28 and a.shape[-2] == 28:
            imgs = a
        elif lab is None and a.ndim == 1 and set(np.unique(a).tolist()) <= set(range(10)):
            lab = a.astype(int)
    if imgs is None or lab is None:
        return {"error": "samples", "msg": f"Could not find 28x28 images and 0-9 labels in {z.files}"}
    imgs = imgs.reshape(-1, 1, 28, 28).astype(np.float32)
    if imgs.max() > 1.5:
        imgs = imgs / 255.0
    best = None
    for name, f in (("0-1 pixels", lambda x: x), ("mean/std normalised", lambda x: (x - 0.1307) / 0.3081)):
        X = torch.from_numpy(f(imgs))
        acc = faults.accuracy(model, X, lab)
        if best is None or acc > best[0]:
            best = (acc, name, X)
    acc, prep, X = best
    return {"model": model, "X": X, "y": lab, "prep": prep, "clean_fp32": acc, "n": len(lab), "img01": imgs[:, 0]}


@st.cache_data(show_spinner="Running the faults...")
def heat_data(fmt, seeds=3):
    import faults
    A = get_assets()
    nb = faults.NBITS[fmt]
    bits = list(range(22, 32)) if fmt == "FP32" else list(range(nb))
    base = faults.accuracy(faults.corrupt(A["model"], fmt, "ber", ber=0.0, want=("base",))["models"]["base"], A["X"], A["y"])
    grid = np.zeros((len(faults.LAYERS), len(bits)))
    for i, layer in enumerate(faults.LAYERS):
        for j, b in enumerate(bits):
            accs = [faults.accuracy(faults.corrupt(A["model"], fmt, "targeted", seed=s, layer=layer, bit=b, want=("none",))["models"]["none"],
                                    A["X"], A["y"]) for s in range(seeds)]
            grid[i, j] = base - float(np.mean(accs))
    return bits, grid, base


@st.cache_data(show_spinner="Running the faults...")
def curve_data(fmt, seeds=5):
    import faults
    A = get_assets()
    bers = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    out = {k: [] for k in ("none", "parity", "nulling")}
    for ber in bers:
        accs = {k: [] for k in out}
        for s in range(seeds):
            r = faults.corrupt(A["model"], fmt, "ber", seed=s, ber=ber, want=tuple(out))["models"]
            for k in out:
                accs[k].append(faults.accuracy(r[k], A["X"], A["y"]))
        for k in out:
            out[k].append((float(np.mean(accs[k])), float(np.min(accs[k])), float(np.max(accs[k]))))
    return bers, out


def _style_ax(ax):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color("#243040")
    ax.tick_params(colors=MUTED, labelsize=8)


def bit_strip(bits_before, bits_after, nb):
    cells = []
    fmt_groups = {32: lambda i: "s" if i == 31 else "e" if i >= 23 else ""}
    for i in range(nb - 1, -1, -1):
        b0, b1 = (bits_before >> i) & 1, (bits_after >> i) & 1
        flipped = b0 != b1
        cls = "bit " + (fmt_groups[32](i) if nb == 32 else ("s" if i == nb - 1 else ""))
        style = "background:#FF5D5D;color:#fff;border-color:#FF5D5D;" if flipped else ""
        cells.append(f'<div class="{cls}" style="{style}">{b1}</div>')
    return (f'<div class="bits" style="grid-template-columns:repeat({nb},minmax(0,1fr));max-width:{min(nb * 34, 760)}px">'
            + "".join(cells) + "</div>")


def explain(text):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)


def title(t):
    st.markdown(f'<div class="ptitle">{t}</div>', unsafe_allow_html=True)


def apply_prep(img01, A):
    """img01: (n,28,28) floats in 0..1 -> model input tensor, using the same preprocessing as the saved digits."""
    import torch
    x = img01.astype(np.float32)
    if A["prep"].startswith("mean"):
        x = (x - 0.1307) / 0.3081
    return torch.from_numpy(x).unsqueeze(1)


def prep_image(file, invert):
    """Any picture -> 28x28 MNIST-style grid (white ink on black, 20x20 digit centred)."""
    from PIL import Image
    im = Image.open(file).convert("L")
    w, h = im.size
    s_ = min(w, h)
    im = im.crop(((w - s_) // 2, (h - s_) // 2, (w - s_) // 2 + s_, (h - s_) // 2 + s_))
    a = np.asarray(im.resize((20, 20), Image.LANCZOS), dtype=np.float32) / 255.0
    inv = a.mean() > 0.5 if invert == "Auto" else invert == "Yes"
    if inv:
        a = 1 - a
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    out = np.zeros((28, 28), dtype=np.float32)
    out[4:24, 4:24] = a
    return out


def _digit_html(labels, truth=None):
    cells = []
    for i, l in enumerate(labels):
        col = TEAL if (truth is None or truth[i] == l) else RED
        cells.append(f'<span style="color:{col}">{l}</span>')
    return '<span class="reads">' + " ".join(cells) + "</span>"


def get_input(A):
    """Returns (img01 array (n,28,28), truth list or None, kind)."""
    kind = st.radio("Input", ["Type a number", "Upload a picture", "Pick a sample digit"], horizontal=True, label_visibility="collapsed")
    if kind == "Type a number":
        txt = st.text_input("Type a number (up to 6 digits)", "2024", max_chars=12)
        digits = [int(c) for c in txt if c.isdigit()][:6]
        skipped = [c for c in txt if not c.isdigit() and not c.isspace()]
        if skipped:
            st.caption(f"Skipped (this network only reads the digits 0 to 9): {' '.join(skipped)}")
        var = st.number_input("Handwriting style", 0, 50, 0, help="Each digit is a real handwritten MNIST sample. Change this number to get different handwriting.")
        if not digits:
            return None, None, kind
        imgs = []
        for pos, d in enumerate(digits):
            idx = np.flatnonzero(A["y"] == d)
            if len(idx) == 0:
                st.caption(f"No saved sample of digit {d}, skipped.")
                continue
            imgs.append(A["img01"][idx[(int(var) + pos) % len(idx)]])
        return (np.stack(imgs), [d for d in digits if (A["y"] == d).any()], kind) if imgs else (None, None, kind)
    if kind == "Upload a picture":
        up = st.file_uploader("Add a picture: a handwritten digit, a flower, a cat, anything", type=["png", "jpg", "jpeg", "webp"])
        inv = st.radio("Invert colours?", ["Auto", "Yes", "No"], horizontal=True,
                       help="The network learned white digits on a black background. Yes = your picture has dark ink on white paper.")
        if up is None:
            return None, None, kind
        return prep_image(up, inv)[None], None, kind
    n = st.slider("Which saved digit?", 0, A["n"] - 1, 0)
    return A["img01"][n][None], [int(A["y"][n])], kind


def story(kind, fmt, mode, layer, bit, ber, prot, info, probs3, n_inputs):
    p0, p1, p2 = probs3
    l0, l1, l2 = p0.argmax(1), p1.argmax(1), p2.argmax(1)
    c0, c1, c2 = p0.max(1), p1.max(1), p2.max(1)
    where = (f"bit {bit} of 2% of the weights in layer {layer}" if mode == "One bit position"
             else f"each stored bit with a chance of {ber:.0e}, in every layer")
    items = [f"<b>Your input became numbers.</b> The picture was shrunk to a 28 &times; 28 grid of brightness values. That grid is all the network ever sees."]
    items.append(f"<b>The network stores 61,706 weights as {fmt} bits.</b> I flipped {where}. In total {info['flipped_bits']} bits flipped, hitting {info['weights_hit']} weights.")
    ex = info["example"]
    if ex:
        items.append(f"<b>Example of damage.</b> One weight in {ex['layer']} changed from <b>{ex['before']:.4g}</b> to <b>{ex['after']:.4g}</b>.")
    read0 = ", ".join(f"{a} ({b * 100:.0f}% sure)" for a, b in zip(l0, c0))
    read1 = ", ".join(f"{a} ({b * 100:.0f}% sure)" for a, b in zip(l1, c1))
    read2 = ", ".join(f"{a} ({b * 100:.0f}% sure)" for a, b in zip(l2, c2))
    items.append(f"<b>What the network said.</b> Before the fault: {read0}. After the fault: {read1}. With {prot.lower() if prot != 'None' else 'no protection'}: {read2}.")
    why = []
    if fmt == "FP32" and mode == "One bit position" and bit >= 23:
        why.append("Bits 23 to 30 of an FP32 number are its exponent, so flipping one can turn a small weight into a gigantic one that drowns everything else.")
    elif fmt != "FP32":
        why.append(f"{fmt} numbers can only hold a small fixed range, so even a flipped top bit cannot create a gigantic weight. That is why damage is usually small.")
    if prot == "Parity-zero":
        why.append("Parity-zero noticed weights where an odd number of bits changed and set them to 0, which is a small, harmless error.")
    if prot == "Weight nulling":
        why.append("Weight nulling set any weight bigger than the largest original weight of its layer to 0." + (" In INT formats this rarely triggers, because values can never exceed that range." if fmt != "FP32" else ""))
    if why:
        items.append("<b>Why.</b> " + " ".join(why))
    if kind == "Upload a picture":
        items.append("<b>Note.</b> This network only knows handwritten digits 0 to 9. For a flower or a cat it just picks the digit that looks closest, so the answer itself is not meaningful. What matters is how the faults change what it says and how sure it is.")
    return '<div class="story-box"><div class="ptitle">What happened, exactly</div><ol>' + "".join(f"<li>{i}</li>" for i in items) + "</ol></div>"


def render():
    import faults
    st.markdown('<a class="back" href="?" target="_self">&larr; BACK</a>'
                '<div class="eyebrow">05 &middot; live demo</div><div class="hero"><h1>Try yourself</h1>'
                '<div class="sub">Give the network a number or any picture, break some stored bits, and see what it says. '
                'Every box has a short note. Simulation only.</div></div>', unsafe_allow_html=True)
    A = get_assets()
    if "error" in A:
        if A["error"] == "missing":
            msg = ("The demo needs two files that come from your Colab notebook. Put <b>" + ", ".join(A.get("need", [])) +
                   "</b> in the same folder as app.py, then restart the app. (Run the export cell in Colab to create them.)")
        else:
            msg = A.get("msg", "Could not load the demo files.")
        st.markdown(f'<div class="stub" style="margin-top:1.4rem">{msg}</div>', unsafe_allow_html=True)
        _verilog_panel()
        return
    if A["clean_fp32"] < 80:
        st.markdown(f'<div class="stub" style="margin:1rem 0">The model loaded, but it only gets {A["clean_fp32"]:.1f}% on the saved digits. '
                    'This usually means the layers in my file differ from your Colab model (for example tanh or average pooling instead of ReLU and max pooling). '
                    'Run <code>print(model)</code> in Colab and send me the output.</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 3.2], gap="large")
    with left:
        with st.container(border=True):
            title("Fault controls")
            fmt = st.radio("Number format", ["FP32", "INT8", "INT4"], index=0, horizontal=True,
                           help="How each weight is stored: 32, 8 or 4 bits.")
            nb = faults.NBITS[fmt]
            mode = st.radio("Fault mode", ["One bit position", "Random bit errors"],
                            help="One bit position: flip the same bit in 2% of one layer. Random: every stored bit may flip.")
            if mode == "One bit position":
                layer = st.selectbox("Layer", faults.LAYERS, index=2)
                bit = st.slider("Bit position (0 = smallest)", 0, nb - 1, 30 if fmt == "FP32" else nb - 1, key=f"bit_{fmt}")
                ber = 0.0
            else:
                layer, bit = "fc1", 0
                ber = st.select_slider("Bit error rate", options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], value=1e-2,
                                       format_func=lambda v: f"{v:.0e}")
            prot = st.radio("Protection", ["None", "Parity-zero", "Weight nulling"], index=1)
            seed = st.number_input("Random seed", 0, 999, 0, help="Changes which weights get hit.")
        st.markdown('<a class="back" style="margin-top:12px" href="?page=limits" target="_self">What this does not show &rarr;</a>',
                    unsafe_allow_html=True)

    mode_key = "targeted" if mode == "One bit position" else "ber"
    res = faults.corrupt(A["model"], fmt, mode_key, seed=int(seed), layer=layer, bit=bit, ber=ber)
    M, info = res["models"], res["info"]
    pkey = {"None": "none", "Parity-zero": "parity", "Weight nulling": "nulling"}[prot]

    with right:
        # ---------------- step 1: your own input
        with st.container(border=True):
            title("Step 1 · Your input")
            img01, truth, kind = get_input(A)
            explain("Type a number (real handwritten digits are used), upload any picture, or pick one saved digit. Whatever you give is turned into a 28 &times; 28 grid, which is what this network reads.")
        if img01 is not None:
            Xin = apply_prep(img01, A)
            P = [faults.probs(M[k], Xin) for k in ("base", "none", pkey)]
            with st.container(border=True):
                title("Step 2 · What the network says")
                n = len(img01)
                cols = st.columns(min(n, 6))
                for i, c in enumerate(cols[:n]):
                    with c:
                        st.image(img01[i], width=84, clamp=True, caption=f"input {i + 1}")
                l0, l1, l2 = (p.argmax(1) for p in P)
                st.markdown(
                    '<div class="readrow"><span class="rl">before fault</span>' + _digit_html(l0, truth) + '</div>'
                    '<div class="readrow"><span class="rl">after fault</span>' + _digit_html(l1, truth) + '</div>'
                    f'<div class="readrow"><span class="rl">{prot.lower() if prot != "None" else "no protection"}</span>' + _digit_html(l2, truth) + '</div>',
                    unsafe_allow_html=True)
                fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.3), facecolor=PANEL, sharey=True)
                for ax, p, name, col in zip(axes, [P[0][0], P[1][0], P[2][0]], ["before fault", "after fault", "protected"], [TEAL, RED, AMBER]):
                    colors = [col if k == p.argmax() else "#2A3A4D" for k in range(10)]
                    ax.bar(range(10), p * 100, color=colors)
                    ax.set_xticks(range(10))
                    ax.set_title(name, color=TEXT, fontsize=9)
                    ax.set_ylim(0, 105)
                    _style_ax(ax)
                axes[0].set_ylabel("how sure (%)", color=MUTED, fontsize=8)
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                explain("Top lines: the digit(s) the network reads, in <span style='color:#3DD6C6'>green</span> when they match the true digit (or when it cannot be checked, like a picture you uploaded). Bars: how sure the network is about each digit 0 to 9, for the <b>first</b> input. Tall wrong bars are the dangerous case: wrong but very sure.")
            st.markdown(story(kind, fmt, mode, layer, bit, ber, prot, info, P, len(img01)), unsafe_allow_html=True)

        # ---------------- step 3: the bigger picture on the saved digits
        X, y = A["X"], A["y"]
        acc_clean = faults.accuracy(M["base"], X, y)
        acc_fault = faults.accuracy(M["none"], X, y)
        acc_prot = faults.accuracy(M[pkey], X, y)
        extra = info["parity_overhead_pct"] if prot == "Parity-zero" else 0.0
        st.markdown('<div class="ptitle" style="margin-top:6px">Step 3 · The bigger picture (all saved digits)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="kpis">'
            f'<div class="kpi"><div class="l">Before any fault ({fmt})</div><div class="v">{acc_clean:.1f}%</div><div class="d">right answers on {A["n"]} saved digits</div></div>'
            f'<div class="kpi bad"><div class="l">After the fault, no protection</div><div class="v">{acc_fault:.1f}%</div><div class="d">{info["flipped_bits"]} bits flipped in {info["weights_hit"]} weights</div></div>'
            f'<div class="kpi good"><div class="l">With: {prot}</div><div class="v">{acc_prot:.1f}%</div><div class="d">{"same as unprotected" if prot == "None" else f"{acc_prot - acc_fault:+.1f} points vs unprotected"}</div></div>'
            f'<div class="kpi"><div class="l">Extra storage</div><div class="v">+{extra:.1f}%</div><div class="d">{"1 parity bit per weight" if prot == "Parity-zero" else "no extra bits needed"}</div></div>'
            '</div>', unsafe_allow_html=True)
        explain("<b>What are these?</b> Your one input above is just one example. These four numbers use all the saved test digits: how good the network is before, how bad after the fault, how much a protection wins back, and what the protection costs in memory.")

        with st.container(border=True):
            title("One weight, bit by bit")
            ex = info["example"]
            if ex:
                st.markdown(bit_strip(ex["bits_before"], ex["bits_after"], ex["nbits"]), unsafe_allow_html=True)
                st.markdown(f'<div class="vals"><span class="ok">before: {ex["before"]:.5g}</span><span class="bad">after: {ex["after"]:.5g}</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="vals">No weight was hit with these settings.</div>', unsafe_allow_html=True)
            explain("One real weight from layer <b>" + (ex["layer"] if ex else layer) + "</b>, as the switches that store it. Red switches flipped. Compare the two numbers: this is the damage one flipped bit does.")

        with st.container(border=True):
            title(f"Which layer and bit hurts most ({fmt})")
            bits, grid, base = heat_data(fmt)
            fig, ax = plt.subplots(figsize=(8.4, 2.9), facecolor=PANEL)
            cmap = LinearSegmentedColormap.from_list("dmg", [PANEL, "#7a2b33", RED])
            im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=0, vmax=max(1.0, float(grid.max())))
            ax.set_xticks(range(len(bits)), [str(b) for b in bits])
            ax.set_yticks(range(len(faults.LAYERS)), faults.LAYERS)
            ax.set_xlabel("bit position" + (" (top 10 bits shown)" if fmt == "FP32" else ""), color=MUTED, fontsize=8)
            _style_ax(ax)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color=TEXT)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label("accuracy lost (points)", color=MUTED, fontsize=8)
            cb.ax.tick_params(colors=MUTED, labelsize=7)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            explain("<b>Heatmap.</b> Each box: flip that one bit in 2% of that layer's weights and see how many accuracy points are lost (average of 3 seeds, saved digits). Darker red = more damage.")

        with st.container(border=True):
            title(f"Random bit errors: accuracy vs error rate ({fmt})")
            bers, out = curve_data(fmt)
            fig, ax = plt.subplots(figsize=(8.4, 2.9), facecolor=PANEL)
            for k, col, lab in (("none", RED, "no protection"), ("parity", TEAL, "parity-zero"), ("nulling", AMBER, "weight nulling")):
                m = np.array([v[0] for v in out[k]]); lo = np.array([v[1] for v in out[k]]); hi = np.array([v[2] for v in out[k]])
                ax.plot(bers, m, "-o", color=col, ms=4, lw=1.6, label=lab)
                ax.fill_between(bers, lo, hi, color=col, alpha=0.15, lw=0)
            ax.set_xscale("log")
            ax.set_xlabel("bit error rate (share of stored bits flipped)", color=MUTED, fontsize=8)
            ax.set_ylabel("accuracy (%)", color=MUTED, fontsize=8)
            ax.set_ylim(0, 100)
            _style_ax(ax)
            leg = ax.legend(fontsize=8, facecolor=PANEL, edgecolor="#243040")
            for t in leg.get_texts():
                t.set_color(TEXT)
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)
            explain("<b>Curve.</b> Flip every stored bit with the chance shown on the bottom, in all layers, and measure accuracy. Lines are the average of 5 seeds, the faint band is the lowest to highest seed. A line that stays high means that protection works at that error rate.")
        _verilog_panel()


def _verilog_panel():
    with st.container(border=True):
        title("The protection as a chip circuit")
        st.code(PARITY_V, language="verilog")
        rows = "".join(f'<tr><td>{k}</td><td>{g}</td><td>{l}</td><td>{p}</td></tr>' for k, (g, l, p) in YOSYS.items())
        st.markdown(f'<div class="tbl"><table><tr><th>Format</th><th>Gates</th><th>LUT4</th><th>vs one multiplier</th></tr>{rows}</table></div>'
                    '<div class="vals"><span class="ok">Icarus exhaustive testbench: ALL TESTS PASSED</span></div>', unsafe_allow_html=True)
        explain("<b>Parity-zero in Verilog</b> (a language that describes chips). <code>err</code> is 1 when an odd number of bits changed; then the weight is read as 0. The table is a rough Yosys size estimate of this checker for each format, not a finished chip.")
