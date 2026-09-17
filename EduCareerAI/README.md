# EduCareer AI

Streamlit education and career assistant with local resume analysis and a small GGUF language model.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The first launch downloads the model from Hugging Face. The model is cached by the runtime and is not stored in this repository.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `.gitignore`, and `README.md`.
2. Open [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Select the repository, branch, and `app.py` as the main file.
4. Click **Deploy** and wait for the first model download to finish.

The app does not require API keys or Streamlit secrets.