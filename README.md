# TechPack Transformer

Django web app that turns a natural-language garment description into a structured tech pack and renders it as an SVG. The model is a small encoder-decoder transformer trained on synthetic data, runs on CPU, and the production checkpoint is included via Git LFS so no training is needed to use the application.

## Quick start (Docker)

```
git clone https://github.com/George-H-L/TechPack-transformer.git
cd TechPack-transformer
docker compose up --build
```

Open `http://localhost:8000`.

First build takes 5-10 minutes (downloads Python image, installs PyTorch CPU build). After that it starts in seconds. Stop with `Ctrl+C`, or `docker compose down` from another terminal. The SQLite database is held in a Docker volume so accounts and saved tech packs persist across restarts.

> **Git LFS required.** The production model weight (`best_model_v3.pth`, 127 MB) is stored in Git LFS. `git clone` fetches it automatically as long as Git LFS is installed (`git lfs install` once, then clone as normal). Without it the weight file will be a text pointer and the app will fail to start.

## Without Docker

```
git clone https://github.com/George-H-L/TechPack-transformer.git
cd TechPack-transformer/TechPackApp
python -m venv venv
```

Activate the venv:
- Windows PowerShell: `venv\Scripts\Activate.ps1`
- macOS/Linux: `source venv/bin/activate`

Install dependencies (the extra index pulls the CPU-only PyTorch wheel):

```
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Run migrations and start the server:

```
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000`. Stop with `Ctrl+C`.

## Operating system

Developed on Windows 11. Runs anywhere Docker or Python 3.12 runs (Windows, macOS, Linux). A GPU is not required.

## Using it

Create an account and click **Create**. Enter a natural-language garment description (e.g. `Oversized boxy tee`). The system either generates the tech pack immediately or asks a follow-up question when confidence is low. From the detail page you can preview the SVG, download it, or open the modify page to adjust measurements. The logo top-left returns you to the home page.

## File layout

```
TechPack-transformer/
├── README.md
├── PROJECTLOG.md
├── FAQ.md
├── docker-compose.yml
└── TechPackApp/
    ├── Dockerfile
    ├── manage.py
    ├── requirements.txt
    ├── eval_compare.py          Pre/post fine-tune comparison
    ├── eval_fields.py           Per-field accuracy
    ├── eval_models.py           Cross-version model comparison
    ├── test_inference.py        Standalone inference harness
    │
    ├── techpack_project/        Django project (settings, URLs, WSGI)
    │
    └── techpack_generator/
        ├── models.py            TechPack DB model
        ├── views.py             Request handlers
        ├── urls.py
        ├── forms.py
        ├── svg_generator.py     Renders tech packs as SVG
        ├── generate_data.py     Claude-based tops data generation
        ├── generate_synthetic_ollama.py  Ollama bottoms data generation
        │
        ├── ml_model/
        │   ├── model.py                  Encoder-decoder transformer architecture
        │   ├── train.py                  Training loop
        │   ├── inference.py              Generation + confidence extraction
        │   ├── tokenizer.py
        │   ├── tokenizer.json            Combined-corpus vocab
        │   ├── tokenizer_tops_only.json  Original tops-only vocab
        │   ├── dataset.py
        │   ├── build_vocab.py
        │   ├── prepare_combined_data.py  Merges Claude tops + Ollama bottoms data
        │   ├── followup.py               Confidence-driven follow-up questions
        │   ├── validation.py             Domain-specific output checks
        │   ├── plot_training.py
        │   └── config.py
        │
        ├── models/
        │   ├── best_model_v3.pth         Production checkpoint (Git LFS, 127 MB)
        │   ├── eval_*.json               Evaluation results
        │   ├── eval_table_pre/post.png   Pre/post fine-tune comparison tables
        │   └── training_curve*.png       Training loss curves
        │
        ├── training_data/
        │   └── schema.json               Data schema (bulk training data not distributed)
        │
        ├── templates/techpack_generator/
        ├── static/
        └── migrations/
```

## What is and isn't in this repo

| Included | Not included |
|---|---|
| Full model source (`model.py`, `train.py`, `inference.py`, …) | Bulk training data (~120 MB of JSON) |
| Tokenizer vocab (`tokenizer.json`) | Experimental checkpoints (v1, v2, v4) |
| Production weight `best_model_v3.pth` via Git LFS | Intermediate training checkpoints |
| Eval results (JSON + PNG) | |
| Training data schema | |

## Dependencies

- Django 4.2.28
- torch 2.7.1 (CPU build)
- numpy 2.1.3
- gunicorn 23.0.0
- whitenoise 6.7.0
