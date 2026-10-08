# Installation, Git publication, Hugging Face deployment, and assignment submission

**Project:** MSAI-631-A01 Group RAG Chatbot • **Team:** Yikebali Mutalipu, Arvind Shankar Raman, Hilal Yilmaz Parlak, Naresh Veluru

Follow the steps in order. The source ZIP is the assignment artifact; a deployed Space is additional evidence, and the results paper is a separate document. **No accounts were accessed or changes published automatically.**

## A. Extract and inspect the submission ZIP

1. Download `Cumberland_RAG_Chatbot_Source.zip` and extract it; do not run Python directly from inside the ZIP.
2. Open the extracted `Cumberland_RAG_Chatbot` folder. Confirm that `app.py`, `rag.py`, `build_index.py`, `requirements.txt`, `README.md`, `data/`, `tests/`, and `docs/` are present.
3. Read `ATTRIBUTION.md` and `docs/LIVE_TEST_CHECKLIST.md` so the group can acknowledge AI assistance and clearly document real observations.
4. If the instructor expects real knowledge-base content, replace the **fictional sample** files under `data/` with *authorized, verified* `.txt` documents. Do not add student records or confidential information.

## B. Run locally on a Windows laptop

1. Install Python **3.11, 64-bit** from https://www.python.org/downloads/ and check the option to add it to PATH. Install Git from https://git-scm.com/downloads if publishing by command line.
2. Open the extracted folder in File Explorer. Click the address bar, type `powershell`, and press Enter.
3. Run:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   python build_index.py
   python app.py
   ```

4. If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` for that terminal session or run the venv Python directly (e.g. `.\.venv\Scripts\python.exe app.py`).
5. Open the localhost address shown in the terminal, commonly http://127.0.0.1:7860. The first execution downloads the public sentence embedding model and Flan-T5-small; allow extra time and a network connection.
6. Ask: `When are demonstration assignments due?` and `How should I ask for a sample extension?`. Check source excerpts, and record **what the app actually says**, even if it is incorrect. Try `What is the weather forecast tomorrow?` to check out-of-scope handling.
7. On any change under `data/`, stop the app and run `python build_index.py` again. The index will be regenerated on an empty fresh deployment; editing a knowledge-base document with an existing index requires rebuilding it.
8. Optional unit tests: `python -m pip install -r requirements-dev.txt` followed by `python -m pytest -q`. Unit tests use mocked ML models and do not substitute for live-model tests.

**On macOS/Linux**, replace the first two lines of the command sequence with `python3.11 -m venv .venv` and `source .venv/bin/activate`; the other commands are the same.

## C. Publish source code to an instructor-accessible GitHub repository

**Recommended easiest version-controlled approach:**

1. Sign into https://github.com and create a **new empty repository**, for example `MSAI631-Group-RAG-Chatbot`. Do not initialize with README, `.gitignore`, or license; these files are already included or intentionally omitted.
2. Open PowerShell in the **extracted project directory**, not its parent.
3. Run the following, replacing `YOUR_GITHUB_USERNAME` with the account that owns the repository:

   ```powershell
   git init
   git branch -M main
   git add .
   git commit -m "Initial group RAG chatbot implementation"
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/MSAI631-Group-RAG-Chatbot.git
   git push -u origin main
   ```

4. GitHub may prompt you to authenticate via browser or Git Credential Manager. Do not paste passwords/tokens into the committed files.
5. Refresh the repository page and verify the complete folder structure and commit history. A **private** repo must explicitly grant your instructor access as a collaborator; a **public** repo is instructor-readable without a login, but check your course's sharing preferences.
6. Keep the repository URL to submit with the ZIP and results paper. In later changes: `git add .`, `git commit -m "Describe change"`, `git push`.

**Alternative (browser only):** Use the new repository's **Add file → Upload files**; upload all source files and preserve directories (`data/`, `tests/`, `docs/`). Check every path before committing. Browser uploads are convenient for small projects but Git commands provide clearer version history.

GitHub reference: https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github

## D. Deploy an interactive chatbot to a free Hugging Face Space

1. Sign in at https://huggingface.co and open https://huggingface.co/new-space.
2. Set a name such as `msai631-group-rag-chatbot`. Choose **Gradio** as the SDK, and the free CPU option if available. Select the visibility required by your course. A public Space is easier for an instructor to view. Free availability, resource limits, and sleeping behavior can change.
3. Once created, open the Space's **Files** section and choose **Add file → Upload files**. Upload the extracted contents, including the root `README.md` YAML metadata, `app.py`, `rag.py`, `build_index.py`, `requirements.txt`, and all `data/*.txt` files. Keep folder paths intact. `README.md` must remain at the root.
4. Use a descriptive commit message such as `Add Gradio RAG chatbot and sample knowledge base`. Hugging Face should install dependencies and build the Space from `app.py`. The first build/start can take several minutes because public weights are downloaded.
5. Open the Space's **App** tab, perform the live tests in `docs/LIVE_TEST_CHECKLIST.md`, and save screenshots of the output and retrieved sources. If it fails, view the **Logs** page for missing modules, memory limits, and first-download failures. Record the actual problem and mitigation in the group paper.
6. Copy the Space URL, usually `https://huggingface.co/spaces/YOUR_USERNAME/msai631-group-rag-chatbot`, into the final submission.

**Alternative for Git users:** Clone the newly created Space Git repository first (to preserve its initial metadata), copy project files into it without copying `.venv`, `index`, `.git`, or caches, then commit and push. Use a supported Hugging Face login; never add access tokens to source files.

Official deployment instructions: https://huggingface.co/docs/hub/spaces-sdks-gradio and https://huggingface.co/docs/hub/spaces-config-reference

## E. Recommended final checks and deliverables

1. **Source ZIP:** Upload `Cumberland_RAG_Chatbot_Source.zip` to the course assignment for *Group Project Source Code*.
2. **Git repository:** Supply a direct link that your instructor can open, with the code, README, authorship disclosure, and commit history.
3. **Results paper:** Upload `Group_Project_Results_APA7.docx` to the course assignment for *Group Project Results* (typically separate from source code). The paper discusses the problem, architecture, technologies, testing, challenges, mitigation, and learning outcomes. Update it with actual deployed test observations before submission.
4. **Space link:** Provide the live URL if successfully deployed, and optionally append screen captures from real testing (never fabricate successful model outputs).
5. **Contribution verification:** The design lists four group members. Agree on the actual work performed by each person before claiming specific individual contributions.
6. **Future presentation:** The handout also requests a later presentation of **at most 20 slides** based on the results paper; this package does not contain that future presentation.

## F. Troubleshooting

- **`py` or `python3.11` not found:** Install the supported Python version and reopen terminal. Use `py --list` to inspect versions on Windows.
- **First model load is slow:** It downloads public pretrained weights and then executes on CPU; permit internet access and storage space.
- **`ModuleNotFoundError`:** Activate the same venv used for installation; rerun `python -m pip install -r requirements.txt`.
- **Stale index warning:** Run `python build_index.py` and restart app.
- **No supported answer:** Make sure a matching fact is present in `data/*.txt`; wording and threshold affect retrieval. An unsupported question is not proof of a system defect.
- **Wrong answer despite correct source:** Small FLAN-T5 can misread context; inspect cited passage, shorten source text, clarify the question, and report errors honestly.
- **HF Space build fails:** Check Logs first; test versions and memory usage, then commit a correction.
- **Git authentication fails:** Use interactive browser/Git Credential Manager, or upload through web interface and preserve source structure.
- **Don't include private data:** Git commits are hard to erase completely; inspect files and `git status` before publishing.

_Last prepared October 7, 2026. Validate any changed provider UI or pricing before uploading._
