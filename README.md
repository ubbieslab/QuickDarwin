# DarwinHub pyDarwin Configuration UI

A standalone Streamlit interface for generating and validating pyDarwin `template.txt`, `tokens.json`, and algorithm-specific `options.json` files. This distribution contains the main UI only; it has no chat, API, Ollama, or CrewAI dependency.

## Requirements

- Python 3.12 or newer
- For running generated models: pyDarwin and NONMEM on the target Windows computer
- Optional visualization: R and `Certara.DarwinReporter`

## Run locally

```bash
git clone <your-repository-url>
cd <your-repository-folder>
python -m venv .venv
```

Activate the environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install and run:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Generated projects are saved under `generated_configs/`, which is intentionally ignored by Git.

## Deploy with Streamlit Community Cloud

1. Upload this folder as the root of a GitHub repository.
2. In Streamlit Community Cloud, select the repository and branch.
3. Set the main file path to `app.py`.
4. Deploy.

The file-generation workflow works in hosted Streamlit. Launching local NONMEM/pyDarwin terminals or DarwinReporter requires running the app on the computer where those tools are installed.
