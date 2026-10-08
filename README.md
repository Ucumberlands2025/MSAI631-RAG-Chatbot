---
title: Course FAQ RAG Chatbot
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.5.1
python_version: 3.11
app_file: app.py
suggested_hardware: cpu-basic
---

# Course FAQ Assistant — Retrieval-Augmented Generation (RAG)

A CPU-friendly, no-paid-API demonstration created for MSAI-631-A01, Artificial Intelligence for Human-Computer Interaction, University of the Cumberlands.

## What it implements

The application reads local plain-text documents (`data/*.txt`), splits them into overlapping passages, embeds each passage with `sentence-transformers/all-MiniLM-L6-v2`, stores vectors in an on-disk NumPy file, retrieves the two closest passages to each question, and asks `google/flan-t5-small` to generate a brief answer grounded in those passages. Each answer includes readable source excerpts. Gradio provides the interactive UI.

**All supplied knowledge-base content is fictional test material.** It is not the real university syllabus, schedule, or policy. Replace `data/` files with verified content before using it for real questions. The app neither reads private course systems nor provides authorized student data.

## Quick start

1. Install Python 3.11 (64-bit) and Git.
2. Open Terminal or PowerShell in the extracted project folder.
3. Create a virtual environment and install requirements:

   Windows PowerShell:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
   macOS/Linux:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
4. Index the source documents: `python build_index.py`.
5. Launch the chatbot: `python app.py`.
6. Open the local browser address printed in the terminal (typically `http://127.0.0.1:7860`).

**Important:** An internet connection is needed on the first run to download the open model weights. After caching the models, generation and retrieval can work offline. First start may be slow on CPU. If source documents change, run `python build_index.py` again before restarting the app.

## Test commands

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The automated unit tests use a deterministic mock encoder and generator to test application logic **without downloading LLM weights**. These tests do NOT prove actual Flan-T5 response accuracy or deployment uptime. For a live test, run the application and record the answers and source excerpts using `docs/LIVE_TEST_CHECKLIST.md`.

## Practical limitations

- Answers can be wrong, even when the closest passages are shown; read the evidence rather than trusting the model blindly.
- A semantic-similarity threshold and word-overlap check reduce, but do not eliminate, incorrect answers.
- No model fine-tuning, persistent user memory, web access, login, or live learning-management-system integration.
- This is intended as a small teaching demonstration, not a production academic advisor.

See **SETUP_AND_SUBMISSION.md** for precise GitHub, Hugging Face Space, ZIP, and paper submission instructions. See **ATTRIBUTION.md** for software/model/research credits and the AI-assistance statement.
