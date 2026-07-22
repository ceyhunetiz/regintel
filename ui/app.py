"""Streamlit UI for the Regulatory Intelligence Assistant.

Run:  streamlit run ui/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components

from regintel.generation.rag import RagPipeline

st.set_page_config(page_title="Regulatory Intelligence Assistant",
                   layout="wide")

# --- Yapı Kredi style theme: navy/black + animated network background ------

_NETWORK_BG = """
<canvas id="net"></canvas>
<style>
  html, body { margin:0; padding:0; background:#060B1A; overflow:hidden; }
  #net { position:fixed; inset:0; width:100vw; height:100vh; }
</style>
<script>
const c = document.getElementById("net"), x = c.getContext("2d");
let W, H, nodes = [];
function resize() {
  W = c.width = window.innerWidth; H = c.height = window.innerHeight;
}
window.addEventListener("resize", resize); resize();
const N = Math.min(90, Math.floor(W * H / 22000));
for (let i = 0; i < N; i++) nodes.push({
  x: Math.random() * W, y: Math.random() * H,
  vx: (Math.random() - .5) * .45, vy: (Math.random() - .5) * .45,
  r: 1.2 + Math.random() * 1.8
});
function step() {
  x.clearRect(0, 0, W, H);
  for (const n of nodes) {
    n.x += n.vx; n.y += n.vy;
    if (n.x < 0 || n.x > W) n.vx *= -1;
    if (n.y < 0 || n.y > H) n.vy *= -1;
  }
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i], b = nodes[j];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      if (d < 150) {
        x.strokeStyle = "rgba(45,111,247," + (0.35 * (1 - d / 150)) + ")";
        x.lineWidth = 1;
        x.beginPath(); x.moveTo(a.x, a.y); x.lineTo(b.x, b.y); x.stroke();
      }
    }
  }
  for (const n of nodes) {
    x.fillStyle = "rgba(0,169,224,0.8)";
    x.beginPath(); x.arc(n.x, n.y, n.r, 0, 7); x.fill();
  }
  requestAnimationFrame(step);
}
step();
</script>
"""

components.html(_NETWORK_BG, height=0)

st.markdown("""
<style>
/* Pin the animation iframe behind everything, full screen */
iframe[title="st.iframe"], div[data-testid="stIFrame"] iframe {
  position: fixed !important;
  inset: 0 !important;
  width: 100vw !important;
  height: 100vh !important;
  z-index: -1 !important;
  border: none !important;
}
/* Let the animation show through the app surfaces */
.stApp { background: transparent !important; }
header[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] {
  background: rgba(6, 11, 26, 0.92) !important;
  border-right: 1px solid rgba(45, 111, 247, 0.25);
}
/* Chat bubbles: translucent navy so text stays readable over the canvas */
div[data-testid="stChatMessage"] {
  background: rgba(11, 21, 48, 0.88);
  border: 1px solid rgba(45, 111, 247, 0.18);
  border-radius: 12px;
}
div[data-testid="stExpander"] {
  background: rgba(11, 21, 48, 0.85);
  border-radius: 10px;
}
div[data-testid="stChatInput"] textarea {
  background: rgba(11, 21, 48, 0.95) !important;
}
h1, h2, h3 { color: #E8EDF7; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline()


pipe = get_pipeline()
available = pipe.store.regulations()

# --- Sidebar logo ----------------------------------------------------------
# If the official logo exists (drop it in as ui/logo.svg), use it.
# Otherwise show an original animated double-spiral motif.
_LOGO_FILE = Path(__file__).parent / "logo.svg"

_SPIRAL_LOGO = """
<div class="rif-logo">
  <svg viewBox="0 0 140 84" width="150" xmlns="http://www.w3.org/2000/svg">
    <g fill="none" stroke="#2D6FF7" stroke-width="5" stroke-linecap="round">
      <path class="horn" d="M 66,76 C 38,74 14,60 12,38 C 10,20 26,8 40,14
               C 52,19 54,36 44,42 C 36,47 28,40 32,32"/>
      <path class="horn horn2" d="M 74,76 C 102,74 126,60 128,38 C 130,20 114,8 100,14
               C 88,19 86,36 96,42 C 104,47 112,40 108,32"/>
    </g>
  </svg>
  <div class="rif-title">RegIntel</div>
</div>
<style>
.rif-logo { text-align:center; padding: 4px 0 2px 0; }
.rif-title {
  color:#E8EDF7; font-size:1.45rem; font-weight:700;
  letter-spacing:0.14em; margin-top:-4px;
}
.horn {
  stroke-dasharray: 260;
  stroke-dashoffset: 260;
  animation: rif-draw 2.2s ease-out forwards, rif-glow 4s ease-in-out 2.2s infinite alternate;
}
.horn2 { animation-delay: 0.35s, 2.55s; stroke:#00A9E0; }
@keyframes rif-draw { to { stroke-dashoffset: 0; } }
@keyframes rif-glow {
  from { filter: drop-shadow(0 0 1px rgba(45,111,247,0.2)); }
  to   { filter: drop-shadow(0 0 7px rgba(45,111,247,0.85)); }
}
</style>
"""

with st.sidebar:
    if _LOGO_FILE.exists():
        st.markdown(
            f'<div class="rif-logo">{_LOGO_FILE.read_text(encoding="utf-8")}'
            f'<div class="rif-title">RegIntel</div></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(_SPIRAL_LOGO, unsafe_allow_html=True)
    st.caption("Source-backed regulatory Q&A — runs fully locally.")

    mode = st.radio("Mode", ["Ask", "Compare regulations"])

    if mode == "Ask":
        reg_filter = st.selectbox("Restrict to regulation",
                                  ["All"] + available)
    else:
        if len(available) < 2:
            st.warning("Comparison needs at least two indexed regulations.")
        reg_a = st.selectbox("Regulation A", available, index=0)
        reg_b = st.selectbox("Regulation B", available,
                             index=min(1, len(available) - 1))

    st.divider()
    st.caption(f"Indexed regulations: {', '.join(available) or 'none — run scripts/ingest.py'}")

# --- Chat persistence -------------------------------------------------------
import json
from datetime import datetime

from regintel import config

CHAT_DIR = config.DATA_DIR / "chats"
CHAT_DIR.mkdir(parents=True, exist_ok=True)


def log_exchange(question: str, answer: str, sources: list) -> None:
    """Append every Q&A to a daily JSONL log on local disk."""
    path = CHAT_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "sources": [s["citation"] for s in sources],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def transcript_markdown(messages: list) -> str:
    lines = [f"# RegIntel chat — {datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in messages:
        role = "**Question**" if m["role"] == "user" else "**Answer**"
        lines += [f"{role}:", m["content"], ""]
        for s in m.get("sources", []):
            lines.append(f"- Source: {s['citation']}")
        lines.append("")
    return "\n".join(lines)


# --- Chat ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    if st.session_state.messages:
        st.download_button(
            "Download this chat (.md)",
            transcript_markdown(st.session_state.messages),
            file_name=f"regintel-chat-{datetime.now():%Y%m%d-%H%M}.md",
        )
        if st.button("New chat"):
            st.session_state.messages = []
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for i, src in enumerate(msg.get("sources", []), 1):
            with st.expander(f"[{i}]  {src['citation']}"):
                st.markdown(src["text"])

question = st.chat_input("Ask a regulatory question…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching regulations…"):
            if mode == "Ask":
                reg = None if reg_filter == "All" else reg_filter
                resp = pipe.ask(question, regulation=reg)
            else:
                resp = pipe.compare(question, reg_a, reg_b)

        st.markdown(resp.answer)
        sources = [{"citation": s.citation, "text": s.text} for s in resp.sources]
        for i, src in enumerate(sources, 1):
            with st.expander(f"[{i}]  {src['citation']}"):
                st.markdown(src["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": resp.answer, "sources": sources})
    log_exchange(question, resp.answer, sources)
