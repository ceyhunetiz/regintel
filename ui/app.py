"""Streamlit UI for the Regulatory Intelligence Assistant.

Run:  streamlit run ui/app.py
"""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components

from regintel.generation.rag import RagPipeline, cited_sources, group_sources

st.set_page_config(page_title="RegIntel — Mevzuat İstihbarat Asistanı",
                   layout="wide")

# --- Giriş (SSO) -------------------------------------------------------
# Kurumun kimlik sağlayıcısı üzerinden OIDC girişi. .streamlit/secrets.toml
# yapılandırılmamışsa st.user.is_logged_in özniteliği hiç var olmaz — bu
# durumda uygulama girişsiz (yerel geliştirme) modda çalışır. Bkz.
# .streamlit/secrets.toml.example.
if hasattr(st.user, "is_logged_in") and not st.user.is_logged_in:
    st.title("RegIntel — Mevzuat İstihbarat Asistanı")
    st.write("Devam etmek için kurumsal hesabınızla giriş yapın.")
    if st.button("Giriş yap", type="primary"):
        st.login()
    st.stop()

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

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" '
    'rel="stylesheet">',
    unsafe_allow_html=True)

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
  border-left: 3px solid #2D6FF7;
}
div[data-testid="stChatInput"] textarea {
  background: rgba(11, 21, 48, 0.95) !important;
}
h1, h2, h3 { color: #E8EDF7; }

/* Comparison tables inside chat answers, styled as cards */
div[data-testid="stChatMessageContent"] table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  border: 1px solid rgba(45, 111, 247, 0.25);
  border-radius: 10px;
  overflow: hidden;
  margin: 0.75rem 0;
}
div[data-testid="stChatMessageContent"] th {
  background: rgba(45, 111, 247, 0.18);
  color: #E8EDF7;
  text-align: left;
  padding: 8px 12px;
  font-weight: 600;
  border-bottom: 1px solid rgba(45, 111, 247, 0.3);
}
div[data-testid="stChatMessageContent"] td {
  padding: 8px 12px;
  border-bottom: 1px solid rgba(45, 111, 247, 0.12);
  vertical-align: top;
}
div[data-testid="stChatMessageContent"] tr:last-child td { border-bottom: none; }
div[data-testid="stChatMessageContent"] tr:hover td { background: rgba(45, 111, 247, 0.06); }

@keyframes rif-spin { to { transform: rotate(360deg); } }

/* Replace Streamlit's default top-right running indicator (which can
   randomly show a different animated icon, incl. an easter-egg running-
   man one) with a single consistent circular spinner in the app's accent
   color. Re-skins the actual icon element in place (rather than adding a
   sibling) so the "Stop" button next to it keeps its normal flex layout. */
div[data-testid="stStatusWidgetRunningIcon"] {
  width: 18px !important;
  height: 18px !important;
  border: 3px solid rgba(45, 111, 247, 0.25);
  border-top-color: #2D6FF7;
  border-radius: 50%;
  background: none !important;
  animation: rif-spin 0.8s linear infinite;
}
div[data-testid="stStatusWidgetRunningIcon"] svg { display: none !important; }

/* Same circular spinner inside the AI's chat bubble while it retrieves
   and generates an answer (replaces Streamlit's default spinner icon,
   re-skinned in place rather than layered on top of it). */
div[data-testid="stSpinnerIcon"] {
  display: inline-block;
  width: 18px;
  height: 18px;
  border: 3px solid rgba(45, 111, 247, 0.25);
  border-top-color: #2D6FF7;
  border-radius: 50%;
  background: none !important;
  animation: rif-spin 0.8s linear infinite;
}
</style>
""", unsafe_allow_html=True)

_ASSISTANT_AVATAR = str(Path(__file__).parent / "ai_logo.png")
_USER_AVATAR = str(Path(__file__).parent / "user_avatar.svg")


@st.cache_resource
def get_pipeline() -> RagPipeline:
    return RagPipeline()


pipe = get_pipeline()
available = pipe.store.regulations()

# --- Sidebar logo ----------------------------------------------------------
# Prefer a real brand asset if one exists (ui/logo.svg, then
# ui/main_logo.png), animated with a wipe-in reveal + glow pulse.
# Falls back to an original animated double-spiral motif if neither exists.
_LOGO_SVG_FILE = Path(__file__).parent / "logo.svg"
_LOGO_RASTER_FILE = Path(__file__).parent / "mainlogocropped.png"

_LOGO_STYLE = """
<style>
.rif-logo { text-align:center; padding: 4px 0 0 0; }
.rif-title {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color:#E8EDF7; font-size:1.45rem; font-weight:700;
  letter-spacing:0.14em; margin-top:-10px;
}
.rif-logo-frame {
  width: 100%; max-width: 280px; aspect-ratio: 552 / 314; margin: 0 auto;
  clip-path: inset(0 100% 0 0);
  animation: rif-reveal 1.1s cubic-bezier(.65,0,.35,1) forwards;
}
.rif-logo-img {
  width: 100%; height: 100%; display: block;
  object-fit: contain; object-position: center;
  animation: rif-glow-img 4s ease-in-out 1.1s infinite alternate;
}
@keyframes rif-reveal { to { clip-path: inset(0 0 0 0); } }
@keyframes rif-glow-img {
  from { filter: drop-shadow(0 0 1px rgba(45,111,247,0.2)) brightness(1); }
  to   { filter: drop-shadow(0 0 14px rgba(45,111,247,0.9)) brightness(1.1); }
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

_SPIRAL_LOGO = f"""
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
{_LOGO_STYLE}
"""


def _animated_raster_logo_html(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"""
<div class="rif-logo">
  <div class="rif-logo-frame">
    <img class="rif-logo-img" src="data:image/png;base64,{b64}" alt="Logo"/>
  </div>
  <div class="rif-title">RegIntel</div>
</div>
{_LOGO_STYLE}
"""

def current_user() -> str:
    """Authenticated user's identity for audit logging and display.

    Falls back to "yerel" when auth isn't configured (secrets.toml
    missing — local dev), since st.user has no email/name attributes
    in that case.
    """
    if hasattr(st.user, "is_logged_in") and st.user.is_logged_in:
        return st.user.get("email") or st.user.get("name") or "bilinmeyen"
    return "yerel"


with st.sidebar:
    if hasattr(st.user, "is_logged_in") and st.user.is_logged_in:
        st.caption(f"Giriş yapan: {current_user()}")
        if st.button("Çıkış yap"):
            st.logout()
        st.divider()

    if _LOGO_SVG_FILE.exists():
        st.markdown(
            f'<div class="rif-logo">{_LOGO_SVG_FILE.read_text(encoding="utf-8")}'
            f'<div class="rif-title">RegIntel</div></div>{_LOGO_STYLE}',
            unsafe_allow_html=True)
    elif _LOGO_RASTER_FILE.exists():
        st.markdown(_animated_raster_logo_html(_LOGO_RASTER_FILE), unsafe_allow_html=True)
    else:
        st.markdown(_SPIRAL_LOGO, unsafe_allow_html=True)
    st.caption("Deneme Sürümü")

    mode = st.radio("Mod", ["Sor", "Mevzuatları karşılaştır"])

    if mode == "Sor":
        reg_filter = st.selectbox("Mevzuata göre filtrele",
                                  ["Tümü"] + available)
    else:
        if len(available) < 2:
            st.warning("Karşılaştırma için en az iki mevzuat indekslenmiş olmalı.")
        reg_a = st.selectbox("Mevzuat A", available, index=0)
        # B's options exclude whatever A is set to: with a single indexed
        # regulation (or a user picking the same one twice) both sides
        # used to resolve to the same instrument, and compare then
        # retrieved every chunk twice under two different source numbers
        # and asked the model to contrast an instrument with itself.
        reg_b_options = [r for r in available if r != reg_a]
        reg_b = st.selectbox("Mevzuat B", reg_b_options) if reg_b_options else None

    st.divider()
    st.caption(f"İndekslenen mevzuatlar: {', '.join(available) or 'yok — scripts/ingest.py çalıştırın'}")

# --- Chat persistence -------------------------------------------------------
import json
from datetime import datetime

from regintel import config

CHAT_DIR = config.DATA_DIR / "chats"
CHAT_DIR.mkdir(parents=True, exist_ok=True)


def log_exchange(question: str, answer: str, sources: list) -> None:
    """Append every Q&A to a daily JSONL log on local disk, tagged with
    the authenticated user — an audit trail of who asked what."""
    path = CHAT_DIR / f"{datetime.now():%Y-%m-%d}.jsonl"
    entry = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "user": current_user(),
        "question": question,
        "answer": answer,
        "sources": [s["citation"] for s in sources],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def transcript_markdown(messages: list) -> str:
    lines = [f"# RegIntel sohbet — {datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in messages:
        role = "**Soru**" if m["role"] == "user" else "**Cevap**"
        lines += [f"{role}:", m["content"], ""]
        for s in m.get("sources", []):
            lines.append(f"- Kaynak: {s['citation']}")
        lines.append("")
    return "\n".join(lines)


_FONT_DIR = Path(__file__).parent / "fonts"


def transcript_pdf(messages: list) -> bytes:
    """Render the chat transcript as a PDF.

    Uses the bundled DejaVu Sans (Unicode TTF) rather than a core PDF
    font — core fonts are latin-1 only, which drops Turkish-specific
    letters (İ ı Ş ş Ğ ğ) that appear throughout Turkish-language
    answers and citations.
    """
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    def line(h: float, text: str) -> None:
        # new_x/new_y reset the cursor to the left margin so the next
        # full-width multi_cell doesn't inherit a squeezed x position.
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("DejaVu", "", str(_FONT_DIR / "DejaVuSans.ttf"))
    pdf.add_font("DejaVu", "B", str(_FONT_DIR / "DejaVuSans-Bold.ttf"))
    pdf.add_font("DejaVu", "I", str(_FONT_DIR / "DejaVuSans-Oblique.ttf"))
    pdf.add_page()

    pdf.set_font("DejaVu", "B", 14)
    line(10, f"RegIntel sohbet - {datetime.now():%Y-%m-%d %H:%M}")
    pdf.ln(2)

    for m in messages:
        pdf.set_font("DejaVu", "B", 11)
        line(8, "Soru:" if m["role"] == "user" else "Cevap:")
        pdf.set_font("DejaVu", "", 11)
        line(6, m["content"])
        for s in m.get("sources", []):
            pdf.set_font("DejaVu", "I", 9)
            line(5, f"Kaynak: {s['citation']}")
        pdf.ln(4)

    return bytes(pdf.output())


# --- Chat ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = _ASSISTANT_AVATAR if msg["role"] == "assistant" else _USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        for src in msg.get("sources", []):
            label = ",".join(str(i) for i in src["indices"])
            with st.expander(f"[{label}]  {src['citation']}"):
                st.markdown(src["text"])

question = st.chat_input("Bir mevzuat sorusu sorun…")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=_USER_AVATAR):
        st.markdown(question)

    with st.chat_message("assistant", avatar=_ASSISTANT_AVATAR):
        with st.spinner("Mevzuatlar taranıyor…"):
            if mode == "Sor":
                reg = None if reg_filter == "Tümü" else reg_filter
                results, stream = pipe.ask_stream(question, regulation=reg)
            elif reg_b is None:
                results, stream = [], iter([
                    "Karşılaştırma için en az iki mevzuat indekslenmiş olmalı."])
            else:
                results, stream = pipe.compare_stream(question, reg_a, reg_b)

        # Stream into a placeholder so the raw text can be REPLACED with
        # the cleaned answer below — previously the streamed text (with
        # markers cited_sources later strips) stayed on screen until the
        # next rerun, so the visible [n] markers didn't match the source
        # list rendered under them.
        answer_box = st.empty()
        with answer_box.container():
            raw_answer = st.write_stream(stream)
        # Only show sources the answer actually cited via [n] markers —
        # `results` is the raw retrieval set, which the model may have
        # partly or entirely discarded (e.g. a refusal cites nothing).
        # cited_sources also strips any dangling marker (a [n] citing a
        # source number outside the retrieved list) from the answer text.
        answer, indices, cited = cited_sources(raw_answer, results)
        answer_box.markdown(answer)
        sources = group_sources(indices, cited)
        for src in sources:
            label = ",".join(str(i) for i in src["indices"])
            with st.expander(f"[{label}]  {src['citation']}"):
                st.markdown(src["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources})
    log_exchange(question, answer, sources)

# Rendered last so it always reflects the exchange that was just appended
# above — rendering this earlier in the script left it one message behind,
# since Streamlit executes top-to-bottom on every rerun.
with st.sidebar:
    if st.session_state.messages:
        st.download_button(
            "Bu sohbeti indir (.md)",
            transcript_markdown(st.session_state.messages),
            file_name=f"regintel-sohbet-{datetime.now():%Y%m%d-%H%M}.md",
        )
        st.download_button(
            "Bu sohbeti indir (.pdf)",
            transcript_pdf(st.session_state.messages),
            file_name=f"regintel-sohbet-{datetime.now():%Y%m%d-%H%M}.pdf",
            mime="application/pdf",
        )
        if st.button("Yeni sohbet"):
            st.session_state.messages = []
            st.rerun()
