# LLM Activation Sparsity Analysis with Oracle Activations: A Research and Experimental Project

> Que perd un LLM quand on rend ses activations sparses ? On demande à un *Activation Oracle*,
> un LLM qui sait lire les activations d'un autre LLM, de nous le dire en langage naturel.

Projet PROCOM de Momo, Raph, Max et Léo.

## L'idée

Les méthodes de **sparsité d'activation** (TEAL, prédicteurs type Deja Vu, coupure après calcul, quantization...)
accélèrent l'inférence en mettant à zéro une partie des activations du LLM. On sait mesurer leur coût
global (perplexité, précision), mais pas **quelle information** disparaît, ni **dans quelles couches**.

Les **Activation Oracles** (Karvonen et al., 2025, [arXiv:2512.15674](https://arxiv.org/abs/2512.15674),
dans la lignée de LatentQA) sont des LLM entraînés à lire les activations d'un modèle et à répondre à des
questions sur ce qu'elles contiennent (« De quel sujet parle le texte ? », « Quelle est la réponse au calcul ? »...).

Notre idée : **interroger l'oracle sur les activations denses puis sur les activations sparses**, et regarder
ce qui change. Si l'oracle ne retrouve plus une information après sparsification, c'est que la méthode l'a effacée.

## Comment on s'y prend

```
prompt ──► LLM cible ──► activations denses ──────────────► oracle ──► réponse dense
                    └──► méthode de sparsité ──► activations sparses ──► oracle ──► réponse sparse
                                                                     comparaison ◄──┘
```

1. **Dataset** : des prompts classés par thème (faits, contexte, calcul, texte prévisible...) et, pour
   chaque thème, des questions à poser à l'oracle.
2. **Extraction** : hooks PyTorch qui récupèrent les activations du LLM cible, couche par couche.
3. **Sparsification** : plusieurs familles de méthodes pour ne pas dépendre d'une seule
   (prédicteurs, coupure post-calcul type TEAL, quantization...).
4. **Interrogation de l'oracle** sur les versions dense et sparse, et vérification de sa fiabilité sur des
   activations sparses, qu'il n'a jamais vues pendant son entraînement.
5. **Analyse** : heatmaps couche × thème de la perte d'information, comparées à des mesures classiques
   (divergence KL entre sorties dense et sparse, activation patching).

**Questions de recherche**
- Toutes les méthodes de sparsité effacent-elles le même type d'information ?
- Quelles couches peut-on rendre sparses sans que l'oracle ne remarque de différence ?
- La perte vue par l'oracle suit-elle les métriques classiques (KL, perplexité) ou révèle-t-elle autre chose ?

Une absence de lien entre l'oracle et les métriques classiques serait aussi un résultat.


## Installation (une seule fois)

L'environnement est géré par **[uv](https://docs.astral.sh/uv/)** : `pyproject.toml` liste les dépendances,
`uv.lock` fige toutes les versions (y compris les dépendances indirectes) et `.python-version` fixe Python 3.11.
Tout le monde a donc exactement le même environnement.

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
  oracle/        # chargement et interrogation de l'Activation Oracle (Max, Léo)
  activations/   # extraction des activations via hooks PyTorch (Max, Léo)
  dataset/       # thèmes de prompts et questions posées à l'oracle (Raph, Momo)
  sparsity/      # méthodes de sparsité : prédicteurs, coupure post-calcul, quantization (Raph, Momo)
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
