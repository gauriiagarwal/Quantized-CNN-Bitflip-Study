"""Content for the 'Know the project' section: level picker, kid level, expert flowchart.
All HTML is built from single-line strings (no blank lines / indents) so Streamlit's
markdown never turns it into a code block."""


def _h(*parts):
    return "".join(parts)


def _mini(title, body):
    return f'<div class="mini"><b>{title}</b><span>{body}</span></div>'


def _box(c, tag, title, inner, extra=""):
    return (f'<div class="fbox" style="--c:{c}{extra}"><div class="fn">{tag}</div>'
            f'<h3>{title}</h3>{inner}</div>')


def _arrow(label=""):
    lab = f"<small>{label}</small>" if label else ""
    return f'<div class="farr">&#9660;{lab}</div>'


def _row(*boxes):
    return '<div class="frow">' + "".join(boxes) + "</div>"


# ---------------------------------------------------------------- picker
def picker():
    return _h(
        '<div class="picker">',
        '<a class="pick" href="?page=know&level=0" target="_self" style="--c:#3DD6C6">',
        '<div class="glyph">0</div><h3>Know like a kid</h3>',
        '<p>A simple story with everyday examples. No maths, no jargon. Start here if this is new.</p>',
        '<div class="tag"><span>story</span><span>&rarr;</span></div></a>',
        '<a class="pick" href="?page=know&level=1" target="_self" style="--c:#FF9F45">',
        '<div class="glyph">1</div><h3>Expert level</h3>',
        '<p>The full flowchart: setup, fault injection, results, fixes, hardware cost and limits.</p>',
        '<div class="tag"><span>flowchart</span><span>&rarr;</span></div></a>',
        '</div>')


# ---------------------------------------------------------------- kid level
def _story(n, title, body, extra=""):
    return (f'<div class="story"><div class="sn">{n}</div><div class="sb"><h3>{title}</h3>'
            f'<p>{body}</p>{extra}</div></div>')


def _bits8(bits, hot=None, label=""):
    cells = []
    for i, ch in enumerate(bits):
        cls = "bit hot" if i == hot else "bit"
        cells.append(f'<div class="{cls}">{ch}</div>')
    return f'<div class="bits b8">{"".join(cells)}</div><div class="cap">{label}</div>'


def kid():
    s = []
    s.append(_story(1, "The network is like a student",
        "Imagine a small student who looks at a picture of a handwritten number and says "
        "&ldquo;that is a 7&rdquo;. Our student is a computer program called a <b>CNN</b>. "
        "It practised on 60,000 handwritten digits (a famous set called <b>MNIST</b>) and now gets "
        "about <b>98 out of 100</b> right."))
    s.append(_story(2, "What does the student remember?",
        "Only <b>numbers</b>. Think of a notebook with 61,706 marks. Each mark says how much to "
        "trust one small clue, like &ldquo;a round loop at the top&rdquo;. These marks are called "
        "<b>weights</b>. All the learning lives in this notebook."))
    s.append(_story(3, "Numbers are stored as tiny switches",
        "A computer cannot write numbers on paper. It keeps them as rows of tiny switches, each "
        "<b>ON (1)</b> or <b>OFF (0)</b>. One switch is called a <b>bit</b>. Here is the number 5 "
        "kept in 8 switches. Watch the first switch flip, like a stray cosmic ray or a hacker "
        "touched it:",
        _bits8("00000101", hot=0, label="5 = 00000101. If the first switch flips, the stored number "
               "is read as <b>&minus;123</b> instead of 5. A small number became a big wrong one.")))
    s.append(_story(4, "Some notebooks are much easier to ruin",
        "Our computer can store each mark in a <b>big fancy format</b> (FP32, 32 switches) or in "
        "a <b>small simple format</b> (INT8 with 8 switches, INT4 with 4). In the fancy format, one "
        "special switch decides how many zeros the number has. Flip it and the mark <b>0.3</b> "
        "turns into <b>1.02 followed by 38 zeros</b>. One wrong mark that huge drowns every other "
        "mark. In my test, one well-chosen flip dropped the student from <b>98 to about 13 "
        "correct out of 100</b>.",
        '<div class="callout">Small formats are like a notebook where marks can only be between '
        '&minus;128 and 127. Even the worst typo cannot become gigantic. In my tests the worst '
        'single flip in INT8 cost only about <b>1 point</b>.</div>'))
    s.append(_story(5, "How do we protect the notebook?",
        "Two cheap tricks I tested. <b>The watchman (parity):</b> keep one extra switch that "
        "counts if the number of ON switches is even. If the count is wrong, the watchman shouts "
        "&ldquo;something changed!&rdquo; and we simply read that mark as 0. "
        "<b>The strict teacher (weight nulling):</b> if a mark is impossibly big, cross it out "
        "and write 0. For the fancy format with many random flips this took accuracy from "
        "<b>9.8 back up to about 95 out of 100</b>, with no extra storage.",
        '<div class="callout">The watchman is also a tiny circuit. I wrote it in Verilog (a chip '
        'design language) and counted its size: about <b>11 small logic blocks</b> for 8-bit '
        'numbers.</div>'))
    s.append(_story(6, "Did I really use radiation?",
        "No. Everything is a <b>computer simulation</b>: I flip the switches myself on purpose "
        "and count what breaks. It is one small network and one kind of picture set, so the "
        "results show a trend, not a law for every network. Saying this clearly is part of "
        "doing it honestly."))
    return _h('<div class="stories">', "".join(s), '</div>',
              '<div class="onel"><b>In one line:</b> a network remembers numbers as switches; one flipped '
              'switch can wreck it in the big format, matters little in small formats, and a cheap '
              'watchman or strict teacher fixes most of it.</div>',
              '<a class="next" href="?page=know&level=1" target="_self">Now see the expert flowchart &rarr;</a>')


# ---------------------------------------------------------------- expert level
def expert():
    b1 = _box("#FF6B6B", "1 &middot; PROBLEM", "Stored weights can silently change",
        "<p>Radiation, ageing or an attacker flips a bit in memory. A neural network keeps all its "
        "knowledge as stored numbers, so one wrong bit may change every answer. Question: "
        "<b>which bit, in which layer, in which number format hurts, and what is the cheapest "
        "protection?</b></p>")
    b2 = _box("#5B9DFF", "2 &middot; SETUP", "Data and model",
        '<ul><li>MNIST digits (+ Fashion-MNIST check)</li>'
        '<li>LeNet-style CNN, 5 weight layers: conv1, conv2, fc1, fc2, fc3</li>'
        '<li>61,706 weights, clean accuracy <span class="ok">98.00%</span></li></ul>')
    b3 = _box("#8B7BFF", "3 &middot; STORE AS BITS", "Three number formats",
        '<div class="wrap">' + _mini("FP32", "32 bits: 1 sign, 8 exponent, 23 mantissa")
        + _mini("INT8", "8 bits, scale per tensor") + _mini("INT4", "4 bits, scale per tensor") + "</div>")
    b4 = _box("#FF9F45", "4 &middot; FAULT INJECTION (simulation only)", "Three ways to break bits",
        '<div class="wrap">'
        + _mini("Targeted", "One bit position flipped in 2% of one layer. Gives the layer &times; bit heatmap.")
        + _mini("Random BER", "Every stored bit flips with probability p (1e-4 &hellip; 1e-2).")
        + _mini("Greedy attacker", "Uses gradients to pick the most damaging bit, one flip at a time.") + "</div>")
    b5 = _box("#C77DFF", "5 &middot; MEASURE", "Accuracy drop, with error bars",
        "<p>10 random seeds, 95% bootstrap confidence intervals, plus confidence analysis: how "
        "often is the network <b>wrong but over 90% sure</b>?</p>")
    b6 = _box("#FFD166", "6 &middot; WHAT I FOUND", "Bit position and format matter most",
        '<div class="wrap">'
        + _mini("FP32: exponent bit 30", 'drops accuracy by <span class="num">55 to 81 points</span> depending on layer. One greedy flip: 98% &rarr; <span class="num">12.8%</span>.')
        + _mini("INT8 / INT4", 'worst single bit is the sign bit: &le; <span class="num">1.21</span> (INT8), &le; <span class="num">1.63</span> (INT4) points.')
        + _mini("Layers differ", "Only 11 of 220 layer &times; bit &times; format cells are clearly harmful (CI low bound above 1 point).") + "</div>")
    b7 = _box("#3DD6C6", "7 &middot; MITIGATIONS COMPARED", "Five low-cost fixes",
        '<div class="wrap">'
        + _mini("M1 Parity-zero", "+1 bit/weight; odd flip detected &rarr; weight read as 0")
        + _mini("M2 Top-k bits", "protect only the k worst bits (ideal assumption)")
        + _mini("M3 Mixed precision", "sensitive layers INT8, rest INT4")
        + _mini("M4 Weight nulling", 'out-of-range weight &rarr; 0; FP32 9.8% &rarr; <span class="ok">~95%</span> at BER 1e-2')
        + _mini("M5 Act. clipping", "clamp activations (Ranger-style)") + "</div>")
    b8 = _box("#4CC9F0", "8 &middot; HARDWARE COST", "Parity checker in Verilog",
        '<p>Module <code>parity_zero</code>, tested exhaustively in Icarus. Yosys estimate for INT8: '
        '<span class="ok">16 gates, 11 LUT4</span>, +12.5% storage. Relative estimate only.</p>')
    b9 = _box("#F4A261", "9 &middot; HONEST LIMITS", "What this does not show",
        "<p>Simulation, not real radiation. One small CNN. Ideal top-k assumption. Nulling can be "
        "bypassed by a sign-flip attacker. Parity in FP32 can make wrong answers more confident.</p>")
    b10 = _box("#7BE495", "CONCLUSION", "In this model, small formats (INT8/INT4) are far less fragile, "
        "and a cheap parity or nulling fix covers most of the rest.",
        "<p>Every number comes from my own simulation (MNIST, LeNet-style CNN, 10 seeds). It "
        "indicates a trend, it does not prove it for all networks.</p>")
    mid = '<div class="farr side">&#9654;</div>'
    return _h('<div class="flow">', _row(b1), _arrow(), _row(b2, mid, b3), _arrow("flip bits on purpose"),
              _row(b4), _arrow(), _row(b5), _arrow(), _row(b6), _arrow("so protect cheaply"),
              _row(b7), _arrow(), _row(b8, b9), _arrow(), _row(b10), '</div>')


def render(level):
    """Return (eyebrow, title, subtitle, body_html, back_href)."""
    if level == "0":
        return ("01 &middot; level 0", "Know like a kid", "A short story. No jargon.", kid(), "?page=know")
    if level == "1":
        return ("01 &middot; level 1", "The whole project, step by step", "Simulation study, start to finish.", expert(), "?page=know")
    return ("01 &middot; two levels", "Know the project", "Pick how deep you want to go.", picker(), "?")


# ---------------------------------------------------------------- limitations (plain page, no cards)
def limits_html():
    from flashcards import LIMITS
    rows = []
    for _, colour, head, bullets in LIMITS:
        lis = "".join(f"<li>{b}</li>" for b in bullets)
        rows.append(f'<div class="lim" style="--c:{colour}"><div class="lh"><span>{head}</span></div><ul>{lis}</ul></div>')
    return '<div class="lims">' + "".join(rows) + "</div>"


# ---------------------------------------------------------------- applications (plain page)
APPS = [
    ("#5B9DFF", "Cars and robots", [
        "Driver-assist cars run neural networks on chips that face heat and vibration. A wrongly stored weight could quietly change what the car &ldquo;sees&rdquo;.",
        "Knowing which bits are dangerous tells engineers where to add protection first."]),
    ("#8B7BFF", "Space and satellites", [
        "Radiation in space flips memory bits far more often than on Earth, and nobody can walk up and fix the chip.",
        "Cheap protection matters here, because every extra bit costs weight, power and money."]),
    ("#FF6B6B", "Medical devices", [
        "Wearables and scanners run small networks on tiny chips. A silently wrong answer is worse than an obvious failure.",
        "These devices run on batteries, so a low-cost check like parity is more realistic than heavy protection."]),
    ("#FFD166", "Phones and small smart devices", [
        "Small chips usually store weights as INT8 or INT4 to save memory. This project shows how fragile each format is.",
        "It also shows where one extra protection bit is worth paying for, and where it is not."]),
    ("#FF9F45", "Data centres and security", [
        "With millions of chips, even rare memory errors happen every day. Attackers can also try to flip weights on purpose.",
        "A quick check that catches a flipped weight and sets it to 0 can stop one bad bit from ruining the answer."]),
    ("#4CC9F0", "Chip design", [
        "The layer-by-bit heatmap works like a guide for designers: protect only the few bits that really matter.",
        "I wrote the parity checker in Verilog and counted its size, so a designer can see the cost in gates before building it."]),
]


def sections_html(data, note=""):
    rows = []
    for colour, head, bullets in data:
        lis = "".join(f"<li>{b}</li>" for b in bullets)
        rows.append(f'<div class="lim" style="--c:{colour}"><div class="lh"><span>{head}</span></div><ul>{lis}</ul></div>')
    foot = f'<div class="note">{note}</div>' if note else ""
    return '<div class="lims">' + "".join(rows) + "</div>" + foot


def apps_html():
    return sections_html(APPS, "This study is a simulation on one small network. These are places where the same idea "
                               "could matter, not systems I tested.")


# ---------------------------------------------------------------- future scope (plain page)
FUTURE = [
    ("#5B9DFF", "Bigger networks", [
        "Repeat the study on larger models such as ResNet and transformers, and on harder datasets than MNIST.",
        "This would show whether the findings hold for the networks used in real cars, phones and cameras."]),
    ("#FF6B6B", "Real hardware tests", [
        "Run the same faults on a real chip (for example an FPGA), and later in a radiation test facility.",
        "This would show how close my simulation is to reality, which is the biggest open question."]),
    ("#FF9F45", "Smarter attackers and realistic faults", [
        "Test stronger attackers, and faults that flip several nearby bits at once, as real memory errors often do.",
        "Security teams could use this to check whether a protection really holds up."]),
    ("#7BE495", "Better protection", [
        "Try real error-correcting codes and choose protection layer by layer, instead of my ideal assumption.",
        "Low-cost chips could then protect only what matters and keep power and area small."]),
    ("#8B7BFF", "More formats and fault-aware training", [
        "Test more formats (FP16, BF16, INT2) and train the network to cope with flipped bits from the start.",
        "New AI chips use many of these formats, so the results would apply directly to them."]),
    ("#4CC9F0", "A tool for chip designers", [
        "Turn the scripts into a small tool that takes a network and says which layers and bits to protect.",
        "Designers could use it early in chip design, before spending time on hardware."]),
]


def future_html():
    return sections_html(FUTURE, "These are ideas for next steps, not results. None of them has been done in this study.")


# ---------------------------------------------------------------- tools used (plain page)
TOOLS = [
    ("#5B9DFF", "Python and PyTorch", [
        "PyTorch holds the small digit-recognition CNN (LeNet-style, 61,706 weights) and runs it on MNIST.",
        "Python scripts flip chosen bits in the stored weights and measure the accuracy afterwards."]),
    ("#7BE495", "NumPy, Pandas and Matplotlib", [
        "NumPy does the bit and number work: converting weights to bits, flipping one, converting back.",
        "Pandas keeps the results tables, and Matplotlib draws the heatmaps and accuracy curves."]),
    ("#FFD166", "Google Colab", [
        "Colab ran the long experiments (10 random seeds, every layer and every bit) without needing a strong laptop.",
        "It also saved the trained model and the result files used by this site."]),
    ("#8B7BFF", "Verilog and Icarus Verilog", [
        "Verilog describes the small parity checker as hardware.",
        "Icarus Verilog runs the testbench, so I could check that the checker catches a flipped bit."]),
    ("#FF9F45", "Yosys", [
        "Yosys synthesises the Verilog and counts the gates and LUT4s, so the hardware cost of the checker is a real number.",
        "That is where the INT8, INT4 and FP32 sizes in this project come from."]),
    ("#4CC9F0", "Streamlit", [
        "Streamlit builds this website and the Try yourself demo in plain Python.",
        "It runs the same fault code as the experiments, so the demo and the report agree."]),
    ("#FF6B6B", "Git and GitHub", [
        "Git keeps the history of the code, and GitHub holds the notebooks, Verilog files, results and this site in one place."]),
]


def tools_html():
    return sections_html(TOOLS, "Everything here is free and open-source or free to use.")


# ---------------------------------------------------------------- why I chose this project (plain page)
# EDIT THIS: replace each bullet with your own words.
WHY = [
    ("#5B9DFF", "Where AI meets hardware", [
        "I study VLSI, and I wanted a project that connects chips to machine learning instead of keeping them separate.",
        "A neural network is only numbers stored in memory, so a memory fault is a natural question to ask."]),
    ("#FF6B6B", "One bit, a big problem", [
        "It surprised me that a single flipped bit can break a whole network in one number format and barely matter in another.",
        "I wanted to find out exactly which bits and layers are dangerous."]),
    ("#7BE495", "A cheap fix I could build", [
        "I liked that the answer could be a small, low-cost circuit, so I could write it in Verilog and measure its size.",
        "It let me do both the AI side and the hardware side in one project."]),
]


def why_html():
    return sections_html(WHY, "")


# ---------------------------------------------------------------- GitHub page
REPO_FILES = [
    ("notebooks/", "Colab notebooks: training, fault injection and the mitigation experiments"),
    ("verilog/", "parity_zero.v and its testbench"),
    ("results/", "CSV files with every measured number, and the Yosys gate counts"),
    ("app.py and other .py files", "this Streamlit website"),
    ("requirements.txt", "the Python packages the site needs"),
    ("README.md", "how to run everything"),
]


def github_html(url):
    if url:
        btn = f'<a class="ghbtn" href="{url}" target="_blank">Open the repository &rarr;</a>'
    else:
        btn = '<div class="note">The repository link will appear here once the repository is public.</div>'
    rows = "".join(f'<div class="ghrow"><code>{n}</code><span>{d}</span></div>' for n, d in REPO_FILES)
    run = ('<div class="lim" style="--c:#3DD6C6"><div class="lh"><span>Run the site yourself</span></div>'
           '<ul><li>Install the packages: <code>pip install -r requirements.txt</code></li>'
           '<li>Start it: <code>streamlit run app.py</code></li></ul></div>')
    return (btn + '<div class="lim" style="--c:#5B9DFF"><div class="lh"><span>What is inside</span></div>'
            + '<div>' + rows + '</div></div>' + run)
