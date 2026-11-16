"""Global CSS for the dark dashboard look (palette from the m3 mockup)."""
from urllib.parse import quote

WM_TEXT = "Gauri Agarwal · Banasthali"

_tile = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='360' height='240'>"
    "<text x='50%' y='50%' text-anchor='middle' transform='rotate(-24 180 120)' "
    "font-family='Inter,Arial,sans-serif' font-size='17' font-weight='600' "
    "fill='#E6EDF3' fill-opacity='0.045'>" + WM_TEXT + "</text></svg>"
)
WM_URI = "data:image/svg+xml;utf8," + quote(_tile)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
:root{--bg:#0E1318;--panel:#151C24;--border:#243040;--text:#E6EDF3;--muted:#8B98A8;
--teal:#3DD6C6;--red:#FF5D5D;--mono:'JetBrains Mono','DejaVu Sans Mono',monospace;}
header[data-testid="stHeader"],[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],
[data-testid="stToolbar"],footer,#MainMenu,[data-testid="stDecoration"]{display:none!important}
html,body,.stApp{background:var(--bg)!important;color:var(--text);font-family:'Inter',sans-serif}
.block-container{max-width:1120px!important;padding:2.2rem 1.6rem 5rem!important}
[data-testid='stHeaderActionElements']{display:none!important}
.stApp a,.stApp a:hover,.stApp a *,.stApp a.box *,.stApp a.back{text-decoration:none!important;border-bottom:none!important}
.bit{min-width:0;padding:0;overflow:visible}.bits{width:100%;max-width:100%;overflow:hidden}

/* fixed watermark: faint tiled text + corner badge, never blocks clicks */
.wm-tile{position:fixed;inset:0;z-index:9998;pointer-events:none;background-image:url("__WM__");background-repeat:repeat}
.wm-badge{position:fixed;right:16px;bottom:14px;z-index:9999;pointer-events:none;
 font:600 12px 'Inter',sans-serif;color:var(--text);opacity:.72;letter-spacing:.3px;
 background:rgba(21,28,36,.88);border:1px solid var(--border);border-radius:999px;padding:6px 13px}
.wm-badge b{color:var(--teal);font-weight:700}

/* hero */
.eyebrow{font:600 12px var(--mono);letter-spacing:2px;color:var(--teal);text-transform:uppercase}
.hero h1{font-size:clamp(30px,4.6vw,50px);line-height:1.1;font-weight:800;margin:.5rem 0 .9rem;letter-spacing:-1px}
.hero h1 em{font-style:normal;color:var(--red)}
.hero .sub{color:var(--muted);font-size:17px;max-width:760px;line-height:1.6}
.author{display:flex;flex-wrap:wrap;gap:10px;margin:1.4rem 0 0}
.chip{border:1px solid var(--border);background:var(--panel);border-radius:999px;padding:6px 14px;font-size:13px;color:var(--muted)}
.chip b{color:var(--text);font-weight:600}

/* bit strip */
.strip-wrap{margin:2.2rem 0 .4rem;background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:20px 22px}
.strip-top{display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;font:13px var(--mono);color:var(--muted)}
.strip-top b{color:var(--text)}
.bits{display:grid;grid-template-columns:repeat(32,minmax(0,1fr));gap:3px;margin:16px 0 8px;perspective:600px}
.bit{aspect-ratio:1/1.25;border-radius:4px;display:flex;align-items:center;justify-content:center;font:600 clamp(8px,1.3vw,13px) var(--mono);
 background:#1B2530;color:#9FB0C3;border:1px solid #2A3A4D}
.bit.s{border-color:#6b5b2a;color:#E8C468}.bit.e{border-color:#2B5C58;color:var(--teal)}
.bit.hot{position:relative;transform-style:preserve-3d;animation:flip 4.4s infinite ease-in-out;border-color:var(--red);box-shadow:0 0 0 0 rgba(255,93,93,.5)}
.bit.hot::after{content:"1";position:absolute;inset:-1px;border-radius:4px;background:var(--red);color:#fff;display:flex;align-items:center;justify-content:center;
 transform:rotateY(180deg);backface-visibility:hidden}
.bit.hot{backface-visibility:hidden}
@keyframes flip{0%,35%{transform:rotateY(0)}50%,85%{transform:rotateY(180deg);box-shadow:0 0 18px 2px rgba(255,93,93,.55)}100%{transform:rotateY(360deg)}}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.vals{display:flex;gap:18px;flex-wrap:wrap;margin-top:14px;font:14px var(--mono)}
.vals .ok{color:var(--teal)}.vals .bad{color:var(--red)}

/* sections */
.h2{font-size:12px;font:600 12px var(--mono);letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin:3rem 0 1rem}
.steps{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.step{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:20px}
.step .n{font:700 12px var(--mono);color:var(--teal)}
.step h3{margin:.35rem 0 .4rem;font-size:18px;font-weight:700}
.step p{margin:0;color:var(--muted);font-size:14.5px;line-height:1.6}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:14px}
.kpi{background:var(--panel);border:1px solid var(--border);border-radius:14px;padding:16px 18px}
.kpi .l{font-size:12px;color:var(--muted)}
.kpi .v{font:700 26px var(--mono);margin:6px 0 2px}
.kpi .d{font-size:12px;color:var(--muted)}
.kpi.bad .v{color:var(--red)}.kpi.good .v{color:var(--teal)}

/* six big boxes */
.boxes{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
a.box{display:flex;flex-direction:column;min-height:210px;background:var(--panel);border:1px solid var(--border);border-radius:18px;padding:24px;color:var(--text);
 transition:transform .18s,border-color .18s,box-shadow .18s}
a.box:hover{transform:translateY(-4px);border-color:var(--teal);box-shadow:0 10px 34px rgba(61,214,198,.13)}
a.box .num{font:700 13px var(--mono);color:var(--teal)}
a.box h3{font-size:24px;font-weight:700;margin:.7rem 0 .5rem;letter-spacing:-.4px}
a.box p{color:var(--muted);font-size:14.5px;line-height:1.55;margin:0;flex:1}
a.box .tag{margin-top:16px;font:600 11px var(--mono);letter-spacing:1px;color:var(--muted);text-transform:uppercase;display:flex;justify-content:space-between}
a.box .tag span:last-child{color:var(--teal);font-size:16px;transition:transform .18s}
a.box:hover .tag span:last-child{transform:translateX(5px)}
.note{margin-top:2.5rem;font-size:12.5px;color:var(--muted);border-top:1px solid var(--border);padding-top:14px;line-height:1.6}

/* stub pages */
.back{display:inline-block;color:var(--teal)!important;font:600 13px var(--mono);margin-bottom:1.5rem}
.stub{background:var(--panel);border:1px dashed var(--border);border-radius:16px;padding:40px;text-align:center;color:var(--muted);margin-top:1.2rem}

@media(max-width:820px){.steps,.boxes{grid-template-columns:1fr}.kpis{grid-template-columns:1fr 1fr}.bits{gap:1px;grid-template-columns:repeat(32,minmax(0,1fr))}.bit{font-size:0!important}.bit::before{content:'';}.bit{border-radius:2px;font-size:6px;border-width:1px;aspect-ratio:1/1.6}.strip-wrap{padding:16px 14px}}

/* know-the-project pages */
.picker{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:2rem}
a.pick{display:flex;flex-direction:column;min-height:300px;background:linear-gradient(135deg,color-mix(in srgb,var(--c) 14%,#121922),#121922 65%);
 border:1px solid color-mix(in srgb,var(--c) 50%,#243040);border-radius:20px;padding:28px;color:var(--text);transition:transform .18s,box-shadow .18s}
a.pick:hover{transform:translateY(-5px);box-shadow:0 12px 40px color-mix(in srgb,var(--c) 22%,transparent)}
.glyph{font:800 96px/1 var(--mono);color:var(--c);width:110px;height:130px;border:2px solid var(--c);border-radius:14px;display:flex;align-items:center;justify-content:center;margin-bottom:18px}
a.pick h3{font-size:28px;margin:0 0 8px;font-weight:700}
a.pick p{color:var(--muted);font-size:15px;line-height:1.6;margin:0;flex:1}
a.pick .tag{margin-top:18px;font:600 11px var(--mono);letter-spacing:1px;text-transform:uppercase;color:var(--muted);display:flex;justify-content:space-between}
a.pick .tag span:last-child{color:var(--c);font-size:18px}
.stories{display:flex;flex-direction:column;gap:14px;margin-top:2rem}
.story{display:flex;gap:18px;background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:22px}
.sn{flex:none;width:42px;height:42px;border-radius:50%;background:rgba(61,214,198,.14);color:var(--teal);font:700 18px/42px var(--mono);text-align:center}
.sb h3{margin:0 0 6px;font-size:20px;font-weight:700}
.sb p{margin:0;color:#B4C0CE;font-size:15.5px;line-height:1.7}
.sb p b{color:#fff}
.bits.b8{grid-template-columns:repeat(8,minmax(0,1fr));max-width:420px;margin:16px 0 6px}
.bits.b8 .bit{font-size:20px!important;aspect-ratio:1/1.15}
.cap{font-size:13px;color:var(--muted);line-height:1.5}.cap b{color:var(--red)}
.callout{margin-top:12px;border-left:3px solid var(--teal);background:rgba(61,214,198,.07);padding:10px 14px;border-radius:0 10px 10px 0;color:#B4C0CE;font-size:14.5px;line-height:1.6}
.onel{margin-top:18px;padding:18px 20px;border:1px solid var(--teal);border-radius:14px;background:rgba(61,214,198,.06);color:#D5DEE8;line-height:1.6}
a.next{display:inline-block;margin-top:18px;color:var(--teal);font:600 14px var(--mono)}
/* expert flowchart */
.flow{margin-top:2rem}
.frow{display:flex;gap:16px;align-items:stretch}
.fbox{flex:1;min-width:0;background:linear-gradient(135deg,color-mix(in srgb,var(--c) 16%,#121922),#121922 60%);
 border:1px solid color-mix(in srgb,var(--c) 55%,#243040);border-left:5px solid var(--c);border-radius:14px;padding:14px 16px}
.fn{font:700 11px var(--mono);letter-spacing:1px;color:var(--c)}
.fbox h3{margin:4px 0 6px;font-size:17px;color:#fff}
.fbox p,.fbox li{margin:0;font-size:13.5px;color:#9FB0C3;line-height:1.55}
.fbox ul{margin:0;padding-left:16px}
.fbox code{font-family:var(--mono);color:#E6EDF3}
.wrap{display:flex;gap:10px;margin-top:10px}
.mini{flex:1;min-width:0;background:rgba(8,12,17,.55);border:1px solid color-mix(in srgb,var(--c) 28%,#243040);border-radius:10px;padding:10px 12px}
.mini b{display:block;font-size:13px;color:#fff;margin-bottom:3px}
.mini>span{display:block;font-size:12.5px;color:#8B98A8;line-height:1.45}
.num{color:var(--red);font-weight:700}.ok{color:var(--teal);font-weight:700}
.farr{text-align:center;color:#8B98A8;font-size:20px;height:32px;line-height:32px}
.farr small{font-size:11px;margin-left:8px;vertical-align:middle}
.farr.side{height:auto;align-self:center;flex:none}
@media(max-width:820px){.picker{grid-template-columns:1fr}.frow,.wrap{flex-direction:column}.farr.side{transform:rotate(90deg)}.story{flex-direction:column;gap:10px}}

/* limitations page */
.lims{display:flex;flex-direction:column;gap:14px;margin-top:2rem}
.lim{display:grid;grid-template-columns:290px 1fr;gap:22px;align-items:center;background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:20px 22px}
.lh span{display:inline-block;font-weight:800;font-size:19px;letter-spacing:-.2px;color:var(--c);background:color-mix(in srgb,var(--c) 18%,transparent);border-left:5px solid var(--c);border-radius:8px;padding:10px 16px}
.lim ul{margin:0;padding-left:20px}
.lim li{color:#C4CFDB;font-size:15.5px;line-height:1.65;margin:4px 0}
@media(max-width:820px){.lim{grid-template-columns:1fr;gap:12px}}

/* try yourself dashboard */
[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);border:1px solid var(--border)!important;border-radius:16px!important}
.ptitle{font:700 12px var(--mono);letter-spacing:1.5px;color:var(--teal);text-transform:uppercase;margin-bottom:8px}
.explain{margin-top:10px;font-size:13.5px;line-height:1.6;color:#9FB0C3;border-left:3px solid var(--teal);padding:2px 0 2px 12px}
.explain b{color:#E6EDF3}.explain code{font-family:var(--mono);color:#E6EDF3}
.tbl table{width:100%;border-collapse:collapse;margin:8px 0;font-size:13.5px}
.tbl th{font:600 11px var(--mono);color:var(--muted);text-align:left;padding:6px 8px;border-bottom:1px solid var(--border)}
.tbl td{padding:6px 8px;border-bottom:1px solid #1d2733;color:#D5DEE8}
[data-testid="stRadio"] label,[data-testid="stSelectbox"] label,[data-testid="stSlider"] label,[data-testid="stNumberInput"] label{font-size:13px!important}

.readrow{display:flex;align-items:center;gap:14px;margin:6px 0}
.rl{width:130px;flex:none;font:600 11px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:1px}
.reads{font:800 34px var(--mono);letter-spacing:6px}
.story-box{background:linear-gradient(135deg,rgba(61,214,198,.10),#121922 70%);border:1px solid #1f5c57;border-left:5px solid var(--teal);border-radius:16px;padding:18px 22px;margin:14px 0}
.story-box ol{margin:6px 0 0;padding-left:20px}
.story-box li{color:#C4CFDB;font-size:14.5px;line-height:1.7;margin:5px 0}.story-box li b{color:#fff}
</style>
<div class="wm-tile"></div>
<div class="wm-badge"><b>Gauri Agarwal</b> &middot; Banasthali</div>
""".replace("__WM__", WM_URI)

CSS += """<style>
.ghbtn{display:inline-block;margin:6px 0 18px;padding:12px 22px;border-radius:10px;background:#3DD6C6;color:#0E1318!important;font-weight:700;text-decoration:none}
.ghrow{display:flex;gap:14px;padding:7px 0;border-bottom:1px solid #243040;font-size:15px}
.ghrow code{min-width:210px;color:#3DD6C6}.ghrow span{color:#C9D3DF}
.lim code{color:#3DD6C6;background:#0E1318;padding:1px 6px;border-radius:5px}
@media(max-width:700px){.ghrow{flex-direction:column;gap:2px}.ghrow code{min-width:0}}
</style>"""
