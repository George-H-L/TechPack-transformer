# Artificial Intelligence For The Purpose of Creating Clothes

Django web app that turns a natural-language garment description into a structured tech pack and renders it as an SVG. The model is a small encoder-decoder transformer trained on synthetic data, runs on CPU, and is included in the repo so no training is needed to use the application.

## Operating system

Developed on Windows 11. Runs anywhere Docker or Python 3.12 runs (Windows, macOS, Linux).

## Required programs

Either:
- **Docker Desktop** - covers everything, one command to run.

Or, if not using Docker:
- **Python 3.12** (3.10 or 3.11 also fine)
- **pip** (ships with Python)

A GPU is not required.

## File layout

```
ghl5/
├── README.md
├── PROJECTLOG.md                Weekly project log
├── FAQ.md
├── docker-compose.yml
└── TechPackApp/
    ├── Dockerfile
    ├── manage.py                Django entry point
    ├── requirements.txt
    ├── db.sqlite3               Created on first run
    ├── eval_compare.py          Pre/post fine-tune comparison
    ├── eval_fields.py           Per-field accuracy
    ├── eval_models.py           Cross-version model comparison
    ├── test_inference.py        Standalone inference harness
    │
    ├── techpack_project/        Django project (settings, URLs, WSGI)
    │
    └── techpack_generator/      Main app
        ├── models.py            TechPack DB model
        ├── views.py             Request handlers
        ├── urls.py              Routes
        ├── forms.py
        ├── admin.py
        ├── tests.py
        ├── svg_generator.py     Renders tech packs as SVG
        ├── generate_data.py     tops generation data with claude
        ├── generate_synthetic_ollama.py    Ollama bottoms data generation
        │
        ├── ml_model/
        │   ├── model.py                 Encoder-decoder transformer
        │   ├── train.py                 Training loop
        │   ├── inference.py             Generation + confidence extraction
        │   ├── tokenizer.py
        │   ├── tokenizer.json           Combined-corpus vocab
        │   ├── tokenizer_tops_only.json Original tops-only vocab
        │   ├── dataset.py
        │   ├── build_vocab.py
        │   ├── prepare_combined_data.py Merges Claude tops + Ollama bottoms
        │   ├── followup.py              Confidence-driven follow-up questions
        │   ├── validation.py            Domain-specific output checks
        │   ├── plot_training.py
        │   └── config.py
        │
        ├── models/                      Trained checkpoints + eval outputs
        │   ├── best_model_combined.pth  Final fine-tuned model (used at runtime)
        │   ├── best_model.pth           Tops-only baseline
        │   ├── best_model_v2/v3/v4.pth  Architecture variants
        │   ├── eval_*.json              Eval results
        │   ├── eval_table_pre/post.png  Pre/post fine-tune tables
        │   └── training_curve*.png      Training loss curves
        │
        ├── training_data/
        │   ├── checkpoint_*.json        Claude-generated tops conversations
        │   ├── combined_train.json      Final training set
        │   ├── combined_val.json        Final validation set
        │   ├── schema.json
        │   └── synthetic/               Ollama-generated bottoms data
        │
        ├── templates/techpack_generator/    HTML templates
        ├── static/                          CSS
        └── migrations/                      Django migrations
```

## Install and run with Docker

```
git clone https://campus.cs.le.ac.uk/gitlab/ug_project/25-26/ghl5.git
cd ghl5
docker compose up --build
```

First build takes 5-10 minutes (downloads Python image, installs PyTorch CPU build and Django). After that it starts in seconds.

Open `http://localhost:8000`.

Stop with `Ctrl+C`, or `docker compose down` from another terminal. The SQLite database is held in a Docker volume so accounts and saved tech packs persist across restarts.

## Install and run without Docker

```
git clone https://campus.cs.le.ac.uk/gitlab/ug_project/25-26/ghl5.git
cd ghl5/TechPackApp
python -m venv venv
```

Activate the venv:
- Windows PowerShell: `venv\Scripts\Activate.ps1`
- macOS/Linux: `source venv/bin/activate`

Install dependencies (the extra index pulls the CPU PyTorch wheel, which is much smaller than the default CUDA build):

```
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

Run migrations and start the server:

```
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000`. Stop with `Ctrl+C`.

## Using it

Once running, create an account and click the Create button. The modify button will not be accesible until there is a tech pack associated with the user. Input natural language prompts (e.g., 'Oversized boxy tee'). The system will either generate the Tech Pack or trigger a follow-up before saving. From the detail page you can preview the SVG, download it, or open the modify page to adjust measurements. The logo in the top left functions as moth logos do in website and allows you to go to home.

## Dependencies

In `TechPackApp/requirements.txt`:

- Django 4.2.28
- torch 2.7.1 (CPU build)
- numpy 2.1.3
- gunicorn 23.0.0 (used by the Docker container)
- whitenoise 6.7.0 (static file serving)

## Notes

The trained model (`best_model_combined.pth`) is committed, so inference works straight after install, no training step required. Training and evaluation scripts are kept in the repo for reference.
