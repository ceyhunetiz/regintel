# Running the app

Quick reference so you don't have to ask each time. Run from the project root:

```bash
cd /Users/ceyhunetiz/Desktop/regintel_tur
```

## 1. Launch the chat UI (the normal way to use it)

```bash
source .venv/bin/activate
streamlit run ui/app.py
```

Opens automatically in your browser, usually at `http://localhost:8501`.
Turkish interface; answers come back in whichever language you ask in.

**Before launching**, check nothing is overriding the model in this terminal:

```bash
echo $REGINTEL_MODEL
```

If that prints anything at all, run `unset REGINTEL_MODEL` first (or just
open a fresh terminal) — `config.py` is hardcoded to `qwen/qwen3-32b`, so
this should normally be empty.

**If `REGINTEL_API_KEY` isn't set** (new terminal windows won't have it
unless it's in your `.zshenv`):

```bash
export REGINTEL_API_KEY=sk-or-...
```

## 2. Stop the app

If you started it in the foreground: `Ctrl+C` in that terminal.

If it's running in the background / a terminal you've lost track of:

```bash
pkill -f "streamlit run ui/app.py"
```

## 3. Ask a one-off question from the terminal (no UI)

```bash
python scripts/ask.py "your question here"
```

## 4. Run the eval suite (checks answer quality against known cases)

```bash
python scripts/eval.py --cases tests/eval_cases_v4.yaml
```

Add `--llm echo` to only check retrieval (fast, free, no live model calls).
Stop it early with:

```bash
pkill -f "scripts/eval.py"
```

## Troubleshooting

- **Turkish question, English answer** — this has happened from time to
  time; if it recurs, note the exact question text so it can be
  reproduced and fixed.
- **A wall of `ModuleNotFoundError: No module named 'torchvision'`** when
  Streamlit starts — harmless. It's Streamlit's dev-mode file watcher
  poking at `transformers`' vision-model submodules, which this
  text-only project never needed. The app is still running underneath;
  confirm with `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501`
  (should print `200`). Silence it if it bothers you:
  ```bash
  streamlit run ui/app.py --server.fileWatcherType none
  ```
