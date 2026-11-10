# PROCOM : sparsité d'activation guidée par un oracle

Projet à 4 : Léo BEAUMONT, Raphaël AZANCOT, Maxime PHILIPON, Mouad CHOUKI. L'environnement est géré par **[uv](https://docs.astral.sh/uv/)** :
`pyproject.toml` liste les dépendances, `uv.lock` fige **toutes** les versions (y compris les dépendances
indirectes) et `.python-version` fixe Python 3.11. Tout le monde a donc exactement le même environnement.

## Installation (une seule fois)

1. Installer uv :
   - macOS / Linux : `curl -LsSf https://astral.sh/uv/install.sh | sh`
   - Windows (PowerShell) : `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
2. Cloner et installer :
   ```bash
   git clone <url-du-depot>
   cd procom
   uv sync          # crée .venv/ avec la bonne version de Python et les versions exactes du lock
   uv run python scripts/check_env.py
   uv run pre-commit install   # formatage automatique avant chaque commit
   ```

uv télécharge lui-même Python 3.11 si besoin. Pas besoin d'activer le venv : préfixer les commandes par `uv run`
(ou `source .venv/bin/activate` si vous préférez).

## torch et CUDA

- **Linux / Windows** : torch est installé depuis l'index PyTorch **CUDA 12.6** (voir `[tool.uv.sources]`
  dans `pyproject.toml`). Fonctionne avec un driver NVIDIA récent (≥ 560) ; sans GPU, torch tourne quand même sur CPU.
- **macOS** : torch standard (CPU + MPS sur puce Apple).
- Pour changer de version CUDA (par ex. `cu128` sur un serveur de l'école), modifier l'URL de l'index **dans
  le dépôt**, relancer `uv lock`, et commiter : ne jamais faire de `pip install torch` à la main.

## Règles d'équipe

- Ajouter une librairie : `uv add <paquet>` (ou `uv add --dev <paquet>` pour un outil), puis **commiter
  `pyproject.toml` et `uv.lock` ensemble**.
- Après un `git pull` qui modifie `uv.lock` : `uv sync`.
- Ne jamais utiliser `pip install` dans le projet.
- Données, activations et poids de modèles ne vont pas dans Git (voir `.gitignore` et `data/README.md`).
- Une branche par tâche, puis pull request vers `main`.

## Structure

```
src/procom/
  oracle/        # oracle : neurones réellement nécessaires pour un prompt 
  activations/   # extraction des activations via hooks PyTorch 
  dataset/       # thèmes de prompts et questions posées à l'oracle 
  sparsity/      # méthodes de sparsité : prédicteurs, coupure post-calcul, quantization 
scripts/         # scripts lançables (uv run python scripts/...)
notebooks/       # explorations (sorties nettoyées automatiquement par nbstripout)
configs/         # fichiers de config YAML des expériences
tests/           # uv run pytest
data/            # local uniquement, non versionné
```

## Commandes utiles

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # formatage
uv run jupyter lab       # notebooks (après `uv add --dev jupyterlab`)
```
