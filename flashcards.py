"""Reusable flashcard deck (flip + Next). Rendered in an iframe so it can use JavaScript."""
import json
import streamlit.components.v1 as components

# (group, colour, term, one-line meaning)
TERMS = [
    ("Basics", "#3DD6C6", "Bit", "A tiny switch that is either 0 (OFF) or 1 (ON)."),
    ("Basics", "#3DD6C6", "Weight", "One remembered number inside the network, like one mark in its notebook."),
    ("Basics", "#3DD6C6", "Bit position", "Which switch in the row it is. Some switches matter far more than others."),
    ("Basics", "#3DD6C6", "Sign bit", "The switch that says whether a number is plus or minus."),
    ("Number formats", "#8B7BFF", "FP32", "A big, delicate format with 32 switches that can hold very large and very small numbers."),
    ("Number formats", "#8B7BFF", "INT8 / INT4", "Small, simple formats with 8 or 4 switches that can hold only a few fixed values."),
    ("Number formats", "#8B7BFF", "Exponent bit", "The FP32 switches that decide how many zeros a number has, so one flip can make it huge."),
    ("Number formats", "#8B7BFF", "Quantization", "Turning big, precise numbers into small, simple ones so they take less memory."),
    ("Network and data", "#5B9DFF", "CNN", "A program that looks at pictures and works out what is in them."),
    ("Network and data", "#5B9DFF", "Layer", "One step of the network; ours has 5 layers that hold weights."),
    ("Network and data", "#5B9DFF", "MNIST", "A famous set of handwritten digit pictures, 0 to 9."),
    ("Network and data", "#5B9DFF", "Accuracy", "Out of 100 pictures, how many the network gets right."),
    ("Faults", "#FF6B6B", "Bit flip", "One switch turning into its opposite, 0 to 1 or 1 to 0."),
    ("Faults", "#FF6B6B", "Fault injection", "Flipping bits on purpose, in a simulation, to see what breaks."),
    ("Faults", "#FF6B6B", "BER (bit error rate)", "Out of all stored switches, the share that flipped by mistake."),
    ("Faults", "#FF6B6B", "Attacker", "Someone who picks the most damaging bit to flip on purpose."),
    ("Measuring", "#FFD166", "Sensitivity", "How easily something breaks: how much accuracy drops when it is hit."),
    ("Measuring", "#FFD166", "Heatmap", "A colour table where each box is a result, and darker red means more damage."),
    ("Measuring", "#FFD166", "Confidence interval", "A range that shows how sure we are of a result; it is the error bar on a chart."),
    ("Measuring", "#FFD166", "Overconfidence", "When the network is wrong but still very sure it is right."),
    ("Protection", "#7BE495", "Parity", "One extra switch that counts the ON switches, so a change gets noticed."),
    ("Protection", "#7BE495", "Weight nulling", "If a weight is impossibly large, cross it out and write 0."),
    ("Protection", "#7BE495", "Mixed precision", "Keep the delicate layers in a bigger format and the rest in a smaller one."),
    ("Hardware", "#4CC9F0", "Verilog", "A language for describing chip circuits; I used it to write the parity checker."),
    ("Hardware", "#4CC9F0", "LUT4 / gate", "Tiny building blocks of a circuit; counting them shows how big the circuit is."),
]

LIMITS = [
    ("Limitations", "#5B9DFF", "Model and data", [
        "I tested only one small network and one simple dataset (MNIST). Bigger networks and harder pictures may behave differently.",
        "I looked at only 3 number formats (FP32, INT8, INT4) and one trained model."]),
    ("Limitations", "#FF6B6B", "Faults", [
        "Everything is a computer simulation. I did not use real radiation or a real chip.",
        "I flipped bits only in stored weights, one position at a time or at random. Real faults often hit many nearby bits at once."]),
    ("Limitations", "#FF9F45", "Attacker", [
        "The greedy attacker is simple, not the smartest one. INT8 may be less robust than it looks.",
        "A clever attacker that only flips the sign bit can get around weight nulling."]),
    ("Limitations", "#7BE495", "Mitigations", [
        "For top-k protection I assumed a protected bit never fails. Real protection costs more.",
        "Parity catches only an odd number of flips and cannot repair anything, it just sets the weight to 0. In FP32 it can even make wrong answers more confident."]),
    ("Limitations", "#4CC9F0", "Hardware and statistics", [
        "Gate and LUT4 counts are only estimates. The checker was tested in simulation, not on an FPGA.",
        "I used only 10 seeds, and many small effects (under 1 point) cannot be told apart from noise."]),
]

_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}html,body{margin:0;background:transparent;color:#E6EDF3;font-family:Inter,'Source Sans 3',Arial,sans-serif}
.layout{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(260px,1fr);gap:20px;padding:4px 2px;align-items:start}
.top{display:flex;justify-content:space-between;align-items:center;font:600 12px 'JetBrains Mono','DejaVu Sans Mono',monospace;color:#8B98A8;margin-bottom:10px}
.bar{height:4px;background:#1B2530;border-radius:4px;overflow:hidden;margin-bottom:16px}.bar i{display:block;height:100%;background:var(--c,#3DD6C6);width:0;transition:width .3s,background .3s}
.scene{perspective:1100px;height:310px;cursor:pointer;outline:none}
.card{position:relative;width:100%;height:100%;transform-style:preserve-3d;transition:transform .6s cubic-bezier(.3,.7,.2,1)}
.card.f{transform:rotateY(180deg)}
.face{position:absolute;inset:0;backface-visibility:hidden;-webkit-backface-visibility:hidden;border-radius:20px;padding:28px;display:flex;flex-direction:column;
 background:linear-gradient(135deg,color-mix(in srgb,var(--c) 16%,#121922),#121922 62%);border:1px solid color-mix(in srgb,var(--c) 55%,#243040);border-left:6px solid var(--c)}
.back{transform:rotateY(180deg)}
.tag{font:700 11px 'JetBrains Mono',monospace;letter-spacing:2px;color:var(--c);text-transform:uppercase}
.term{flex:1;display:flex;align-items:center;font-size:clamp(34px,5.4vw,62px);font-weight:800;letter-spacing:-1px;line-height:1.1}
.mean ul{margin:0;padding-left:20px}.mean li{margin:0 0 12px;font-size:clamp(16px,2vw,21px);line-height:1.5;color:#D5DEE8}.mean b.bh{display:block;margin-bottom:10px;font-size:clamp(18px,2.4vw,24px);color:#fff}
.mean{flex:1;display:flex;align-items:center;font-size:clamp(19px,2.6vw,28px);line-height:1.45;color:#E6EDF3;font-weight:500}
.hint{font:12px 'JetBrains Mono',monospace;color:#6B7A8C}
.btns{display:flex;justify-content:space-between;align-items:center;margin-top:16px}
button{font:600 12px 'JetBrains Mono',monospace;letter-spacing:.5px;background:#151C24;color:#E6EDF3;border:1px solid #243040;border-radius:999px;padding:8px 16px;cursor:pointer}
button:hover{border-color:var(--c);color:var(--c)}
button.next{background:var(--c);color:#0E1318;border-color:var(--c)}button.next:hover{color:#0E1318;filter:brightness(1.1)}
.layout.one{grid-template-columns:minmax(0,1fr);max-width:780px}.layout.one .side{display:none}
.side{background:#151C24;border:1px solid #243040;border-radius:16px;padding:16px 16px 8px;overflow:auto}
.side h4{margin:0 0 12px;font:700 11px 'JetBrains Mono',monospace;letter-spacing:2px;color:#8B98A8}
.g{margin-bottom:10px}.g small{display:block;font:700 10px 'JetBrains Mono',monospace;color:var(--gc);letter-spacing:1.5px;margin-bottom:5px;text-transform:uppercase}
.chip{display:inline-block;font:500 12.5px Inter,Arial,sans-serif;padding:4px 10px;border-radius:999px;border:1px solid #2A3A4D;color:#9FB0C3;margin:0 4px 5px 0;cursor:pointer;background:transparent}
.chip:hover{color:#fff;border-color:var(--gc)}
.chip.v{border-color:color-mix(in srgb,var(--gc) 70%,#243040);color:#E6EDF3}
.chip.cur{background:var(--gc);color:#0E1318;font-weight:700;border-color:var(--gc)}
@media(max-width:700px){.layout{grid-template-columns:1fr}.side{display:none}}
</style></head><body><div class="layout" id="app" style="--c:#3DD6C6">
<div><div class="top"><span id="grp"></span><span id="cnt"></span></div>
<div class="bar"><i id="bar"></i></div>
<div class="scene" id="scene" tabindex="0"><div class="card" id="card">
<div class="face"><div class="tag" id="tag1"></div><div class="term" id="term"></div><div class="hint">click the card to flip</div></div>
<div class="face back"><div class="tag" id="tag2">meaning</div><div class="mean" id="mean"></div><div class="hint">click to flip back</div></div>
</div></div>
<div class="btns"><button id="prev">&larr; Prev</button><button id="shuf">Shuffle</button><button class="next" id="next">Next &rarr;</button></div></div>
<div class="side"><h4>ALL TERMS &middot; CLICK TO JUMP</h4><div id="idx"></div></div>
</div><script>
const ORIG=__DATA__;const INDEX=__INDEX__;if(!INDEX)document.getElementById('app').classList.add('one');
function esc(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function setMean(c){const m=c[3];if(Array.isArray(m)){$('mean').innerHTML='<div><b class="bh">'+esc(c[2])+'</b><ul>'+m.map(x=>'<li>'+esc(x)+'</li>').join('')+'</ul></div>'}else{$('mean').textContent=m}}
let D=ORIG.slice(),i=0;const seen=new Set();
const $=id=>document.getElementById(id),card=$('card');
function buildIdx(){let h='',g='';ORIG.forEach(c=>{if(c[0]!==g){if(g)h+='</div>';g=c[0];h+='<div class="g" style="--gc:'+c[1]+'"><small>'+g+'</small>'}
 h+='<span class="chip" data-t="'+c[2].replace(/"/g,'&quot;')+'">'+c[2]+'</span>'});$('idx').innerHTML=h+'</div>'}
function mark(){document.querySelectorAll('.chip').forEach(e=>{const t=e.dataset.t;e.classList.toggle('cur',t===D[i][2]);e.classList.toggle('v',seen.has(t)&&t!==D[i][2])})}
function show(){const c=D[i];seen.add(c[2]);$('app').style.setProperty('--c',c[1]);$('grp').textContent=c[0];$('tag1').textContent=c[0];
 $('term').textContent=c[2];setMean(c);$('cnt').textContent=(i+1)+' / '+D.length;$('bar').style.width=((i+1)/D.length*100)+'%';mark()}
function go(d,abs){const wasF=card.classList.contains('f');card.classList.remove('f');setTimeout(()=>{i=abs!==undefined?abs:(i+d+D.length)%D.length;show()},wasF?280:0)}
$('scene').onclick=()=>card.classList.toggle('f');
$('next').onclick=()=>go(1);$('prev').onclick=()=>go(-1);
$('shuf').onclick=()=>{D=D.map(x=>[Math.random(),x]).sort((a,b)=>a[0]-b[0]).map(x=>x[1]);i=0;card.classList.remove('f');show()};
$('idx').onclick=e=>{const t=e.target.dataset&&e.target.dataset.t;if(t){go(0,D.findIndex(c=>c[2]===t))}};
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')go(1);else if(e.key==='ArrowLeft')go(-1);else if(e.key===' '||e.key==='Enter'){e.preventDefault();card.classList.toggle('f')}});
buildIdx();show();
</script></body></html>"""


def deck(cards, height=560, index=True):
    """cards: list of (group, colour, term, meaning)."""
    data = json.dumps(cards).replace("</", "<\\/")
    components.html(_HTML.replace("__DATA__", data).replace("__INDEX__", "true" if index else "false"), height=height)
