"""Bit-flip project site: landing page (dark dashboard design).
Landing page is pure HTML/CSS -- no torch import, loads instantly."""
import struct
import streamlit as st
from style import CSS
import know
import flashcards

# ---------------- CONFIG ----------------
PROJECT_TITLE = "When one bit flips"
FULL_TITLE = ("Bit-Position and Layer Sensitivity of Quantized CNNs under Weight "
              "Bit-Flip Faults, with a Low-Cost Hardware-Aware Mitigation")
AUTHOR = "Gauri Agarwal"
COLLEGE = "Banasthali Vidyapith"
PROGRAMME = "B.Tech, VLSI"
PAGE_ICON = "⚡"
GITHUB_URL = "https://github.com/gauriiagarwal/Quantized-CNN-Bitflip-Study"
# real numbers from my own simulation (MNIST, LeNet-style CNN, 10 seeds)
KPIS = [
    ("bad",  "One FP32 bit flip (greedy attacker)", "98% → 12.8%", "accuracy after 1 flip"),
    ("bad",  "Worst INT8 single bit", "1.2 pts", "damage, vs 54–81 pts in FP32"),
    ("good", "FP32 at 1% bit errors, weight nulling", "9.8% → ~95%", "free: no extra storage"),
    ("good", "Parity checker (INT8)", "11 LUT4", "16 gates, +12.5% storage"),
]
BOXES = [
    ("know",    "01", "Know the project", "What a bit flip is, what I broke on purpose, and what I found. The whole story, from simple to expert.", "two levels"),
    ("tools",   "02", "Tools used", "Python, PyTorch, Colab, Verilog, Icarus, Yosys and Streamlit: what each one was used for.", "read"),
    ("terms",   "03", "Basic terms", "FP32, INT8, INT4, bit, parity, BER, MNIST: every word, in simple language.", "flashcards"),
    ("apps",    "04", "Applications", "Where a flipped bit really hurts: cars, satellites, medical devices, data centres.", "read"),
    ("try",     "05", "Try yourself", "Type a number or add any picture, break some stored bits, and watch what the network says.", "live demo"),
    ("future",  "06", "Future scope", "Bigger models, real hardware fault injection, smarter protection.", "read"),
    ("limits",  "07", "Limitations", "What this study does not show. Simulation only, one small CNN, ideal assumptions.", "read"),
    ("why",     "08", "Why I chose this project", "The story behind the topic, in my own words.", "read"),
    ("github",  "09", "GitHub", "All code, results and the Verilog checker in one repository.", "link"),
]
PAGES = {b[0]: b for b in BOXES}

st.set_page_config(page_title=f"{PROJECT_TITLE} · {AUTHOR}", page_icon=PAGE_ICON,
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)


def flip_strip():
    """Real FP32 bits of 0.3, with bit 30 flipped."""
    b = struct.unpack(">I", struct.pack(">f", 0.3))[0]
    bad = struct.unpack(">f", struct.pack(">I", b ^ (1 << 30)))[0]
    cells = []
    for i in range(31, -1, -1):
        cls = "s" if i == 31 else "e" if i >= 23 else ""
        if i == 30:
            cls += " hot"
        cells.append(f'<div class="bit {cls}">{(b >> i) & 1}</div>')
    return b, bad, "".join(cells)


def box_html(k, n, h, d, t):
    if k == "github" and GITHUB_URL:
        href, tgt = GITHUB_URL, "_blank"
    else:
        href, tgt = f"?page={k}", "_self"
    arrow = "&nearr;" if tgt == "_blank" else "&rarr;"
    return (f'<a class="box" href="{href}" target="{tgt}" rel="noopener"><div class="num">{n}</div><h3>{h}</h3>'
            f'<p>{d}</p><div class="tag"><span>{t}</span><span>{arrow}</span></div></a>')


def landing():
    b, bad, cells = flip_strip()
    st.markdown(f"""
<div class="hero">
  <div class="eyebrow">Simulation study &middot; {PROGRAMME}</div>
  <h1>When <em>one bit</em> flips, a neural network can break.</h1>
  <div class="sub">{FULL_TITLE}.</div>
  <div class="author"><span class="chip">by <b>{AUTHOR}</b></span><span class="chip"><b>{COLLEGE}</b></span>
  <span class="chip">MNIST &middot; FP32 / INT8 / INT4</span></div>
</div>

<div class="strip-wrap">
  <div class="strip-top"><span>One stored weight, <b>0.3</b>, as 32 bits (FP32)</span><span>hex <b>0x{b:08X}</b></span></div>
  <div class="bits">{cells}</div>
  <div class="legend"><span><i style="background:#E8C468"></i>sign</span><span><i style="background:#3DD6C6"></i>exponent</span><span><i style="background:#9FB0C3"></i>mantissa</span><span><i style="background:#FF5D5D"></i>bit 30 flips</span></div>
  <div class="vals"><span class="ok">before: 0.3</span><span class="bad">after: {bad:.3e}</span></div>
</div>

<div class="h2">The project in one minute</div>
<div class="steps">
  <div class="step"><div class="n">STEP 1</div><h3>A network is just numbers</h3>
   <p>A small CNN reads handwritten digits (MNIST). It stores 61,706 numbers called weights, and each weight is saved as bits (0s and 1s) in memory.</p></div>
  <div class="step"><div class="n">STEP 2</div><h3>Break one bit</h3>
   <p>Radiation, ageing or an attacker can flip a stored bit. I flip bits on purpose, in FP32, INT8 and INT4, and measure how much accuracy is lost and which bit hurts most.</p></div>
  <div class="step"><div class="n">STEP 3</div><h3>Protect it cheaply</h3>
   <p>I compare low-cost fixes (parity bit, protecting only the top bits, weight nulling, clipping) and check the cost of a parity checker written in Verilog.</p></div>
</div>
<div class="kpis">
  {''.join(f'<div class="kpi {k}"><div class="l">{l}</div><div class="v">{v}</div><div class="d">{d}</div></div>' for k, l, v, d in KPIS)}
</div>

<div class="h2">Explore</div>
<div class="boxes">
  {''.join(box_html(*b) for b in BOXES)}
</div>

<div class="note">Simulation only: no real hardware was irradiated. One small CNN on MNIST (plus Fashion-MNIST check),
10 random seeds, 95% bootstrap confidence intervals. Hardware cost numbers are relative Yosys estimates.</div>
""", unsafe_allow_html=True)


def know_page():
    eyebrow, title, sub, body, back = know.render(st.query_params.get("level", ""))
    st.markdown(f'<a class="back" href="{back}" target="_self">&larr; BACK</a>'
                f'<div class="eyebrow">{eyebrow}</div><div class="hero"><h1>{title}</h1>'
                f'<div class="sub">{sub}</div></div>' + body, unsafe_allow_html=True)


def terms_page():
    _, n, head, desc, tag = PAGES["terms"]
    st.markdown('<a class="back" href="?" target="_self">&larr; BACK</a>'
                f'<div class="eyebrow">{n} &middot; {tag}</div><div class="hero"><h1>{head}</h1>'
                f'<div class="sub">25 words, one line each. Click a card to flip it, then press Next.</div></div>',
                unsafe_allow_html=True)
    flashcards.deck(flashcards.TERMS)


def limits_page():
    _, n, head, desc, tag = PAGES["limits"]
    st.markdown('<a class="back" href="?" target="_self">&larr; BACK</a>'
                f'<div class="eyebrow">{n} &middot; {tag}</div><div class="hero"><h1>{head}</h1>'
                f'<div class="sub">What this study does not show.</div></div>',
                unsafe_allow_html=True)
    st.markdown(know.limits_html(), unsafe_allow_html=True)


def apps_page():
    _, n, head, desc, tag = PAGES["apps"]
    st.markdown('<a class="back" href="?" target="_self">&larr; BACK</a>'
                f'<div class="eyebrow">{n} &middot; {tag}</div><div class="hero"><h1>{head}</h1>'
                '<div class="sub">Where one flipped bit could really matter.</div></div>' + know.apps_html(),
                unsafe_allow_html=True)


def future_page():
    _, n, head, desc, tag = PAGES["future"]
    st.markdown('<a class="back" href="?" target="_self">&larr; BACK</a>'
                f'<div class="eyebrow">{n} &middot; {tag}</div><div class="hero"><h1>{head}</h1>'
                '<div class="sub">What I would do next, and where it would help.</div></div>' + know.future_html(),
                unsafe_allow_html=True)


def simple_page(key, sub, body):
    _, n, head, desc, tag = PAGES[key]
    st.markdown('<a class="back" href="?" target="_self">&larr; BACK</a>'
                f'<div class="eyebrow">{n} &middot; {tag}</div><div class="hero"><h1>{head}</h1>'
                f'<div class="sub">{sub}</div></div>' + body, unsafe_allow_html=True)


def try_page():
    import tryit  # imports torch lazily, only on this page
    tryit.render()


def stub(key):
    _, n, head, desc, tag = PAGES[key]
    st.markdown(f'<a class="back" href="?" target="_self">&larr; BACK TO HOME</a>'
                f'<div class="eyebrow">{n} &middot; {tag}</div><div class="hero"><h1>{head}</h1><div class="sub">{desc}</div></div>'
                f'<div class="stub">This page is coming next. Tell me what you want here.</div>',
                unsafe_allow_html=True)


page = st.query_params.get("page", "")
if page == "know":
    know_page()
elif page == "try":
    try_page()
elif page == "future":
    future_page()
elif page == "apps":
    apps_page()
elif page == "limits":
    limits_page()
elif page == "terms":
    terms_page()
elif page == "tools":
    simple_page("tools", "What each tool did in this project.", know.tools_html())
elif page == "why":
    simple_page("why", "The story behind the topic.", know.why_html())
elif page == "github":
    simple_page("github", "All code, results and the Verilog checker in one place.", know.github_html(GITHUB_URL))
elif page in PAGES:
    stub(page)
else:
    landing()
