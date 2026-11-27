# PROCOM : analyser la sparsité d'un LLM avec des Activation Oracles

*Fichier de contexte pour Claude Code, état au 27/11/2026. Peut servir de base à un `CLAUDE.md` à la racine du dépôt. **Mis à jour avant chaque push** : la carte (§0) et le journal (§6) doivent suffire pour savoir quoi ouvrir, sans relire tout le dépôt.*

Équipe : Léo Beaumont, Maxime Philippon (Max), Mouad Chouki (Momo), Raphaël Azancot (Raph). Encadrant : Bastien Pasdeloup. Fin du projet : mars 2027 (revue 2 en S16, revue 3 en S25).

Dépôt : https://github.com/2MOData/LLM-Activation-Sparsity-Analysis-with-Oracle-Activations-A-Research-and-Experimental-Project (README et code en anglais, une branche par tâche puis PR vers `main`).

---

## 0. Carte du dépôt (à lire avant d'ouvrir un fichier)

| Chemin | Rôle | Symboles clés / à savoir |
|---|---|---|
| `src/procom/oracle/aobench.py` | AObench + oracle | `ask_oracle`, `shifted_window`, `run_sweep`/`judge_sweep` (reprise), `LocalJudge`, `collect_results`, `plot_sweep` |
| `src/procom/{dataset,sparsity}/` | vides (docstrings périmées : « predictors, quantization ») | à remplir |
| `src/procom/activations/` | vide, **ignoré par Git** (règle `activations/` de `.gitignore`) | corriger la règle en `/activations/` avant d'y coder |
| `scripts/aobench_layer_sweep.py` + `scripts/slurm/` | exp. 1 sur cluster | modèle de script : argparse, `--limit`, `--prepare`, reprise |
| `notebooks/tutorial_colab_oracle.ipynb` | tuto Colab, **gabarit de style des notebooks** | anglais, badge Colab, §0 Setup identique, hooks retirés dans `finally` |
| `notebooks/aobench_layer_window_sweep.ipynb` | exp. 1 en notebook | = script ci-dessus |
| `notebooks/sparsity_rates_teal_wina.ipynb` | démonstrateur : lire la sparsité réalisée TEAL / WINA sur 2 exemples | voir §3 ; `neuron_scores`, `calibrate_thresholds`, `run_sparse` |
| `results/aobench_layer_sweep/` | résultats exp. 1 | `README.md` résume tout, CSV + figures |
| `tests/` | `test_aobench.py` (utilitaires purs), `test_imports.py` | `uv run pytest` |

Sorties locales dans `outputs/` (ignoré par Git).

---

## 1. Le but du projet

### Le constat
Les méthodes de **sparsité d'activation** (TEAL, WINA...) accélèrent un LLM en mettant à zéro les activations jugées peu utiles. On mesure leur coût avec des métriques de **sortie** : perplexité, KL entre sorties dense et sparse, exactitude. Ces métriques ne disent pas **quelle information interne** est perdue, ni **à quelle couche**.

### L'idée
Un **Activation Oracle** (Karvonen et al., [arXiv:2512.15674](https://arxiv.org/abs/2512.15674)) est un LLM (un LoRA sur le modèle cible lui-même) entraîné à lire le flux résiduel d'un modèle et à répondre en langage naturel à des questions sur son contenu.

On interroge l'oracle sur les activations **denses**, puis sur les activations **sparses** (après coupure), et on compare. Si l'oracle ne retrouve plus une information après la coupure, c'est que la méthode l'a effacée.

```
prompt ──► Qwen3-8B ──► activations denses ─────────────────► oracle ──► réponse dense
                   └──► hook de sparsité ──► activations sparses ──► oracle ──► réponse sparse
                                                                    comparaison ◄──┘
```

### Les trois questions de recherche (version de la présentation S9)
- **RQ1 : des pertes « silencieuses » ?** La sortie reste correcte, mais l'oracle perd-il les signaux métacognitifs (`missing_info`, `backtracking`, `sycophancy`) ou les étapes intermédiaires d'un raisonnement multi-sauts ?
- **RQ2 : où naît l'erreur ?** Quand le modèle coupé se trompe, le concept a-t-il été effacé dans les couches médianes, ou est-ce la formulation qui se dégrade en fin de réseau ?
- **RQ3 : que contient ce qui est coupé ?** L'oracle peut-il décrire en langage naturel Δv = v_dense − v_sparse, l'information portée par les petites activations supprimées ?

Une absence de lien entre ce que voit l'oracle et les métriques classiques serait aussi un résultat.

### Livrables visés
1. Code modulaire (module oracle et AObench, hook commun sparsité / ablation, environnement reproductible, tests).
2. Jeu de données par thèmes, avec activations d'origine et coupées enregistrées.
3. Cartes couche × thème des pertes, croisées avec KL / perplexité ; rapport et article (résultats négatifs compris).
4. Démonstrateur : notebook où l'on entre une phrase, choisit la sparsité par couche et compare les réponses de l'oracle.

---

## 2. Les choix techniques figés

### Modèle cible et oracle
- **Qwen3-8B** (`Qwen/Qwen3-8B`) : 36 couches, dimension 4 096, FFN SwiGLU de 12 288 neurones (gate, up, down), GQA, modèle dense. En bf16 (~16 Go). **Ne pas charger en 4 ou 8 bits** : la quantification modifie elle-même les activations.
- **Oracle** : `ceselder/qwen3-8b-ao-v3-best` (papier *Building Better Activation Oracles*, [arXiv:2606.02609](https://arxiv.org/abs/2606.02609)). Il lit **5 couches consécutives à la fois** et a été entraîné sur les couches **21-25**.
- L'injection dans l'oracle renormalise le vecteur (h' = h + ‖h‖ · v/‖v‖) : l'oracle ne voit que la **direction** des activations, pas leur norme. Il faut donc aussi mesurer la similarité cosinus dense / sparse.
- L'oracle et le modèle cible partagent les mêmes poids de base : **le hook de sparsité doit être retiré** (et le LoRA désactivé pour lire la cible) avant le passage de l'oracle.

### Benchmark de l'oracle : AObench
- Dépôt `ceselder/cot-oracle`, commit figé `bec6f8a82c6882faa3a931a88978bd6af0694c27` (cloné par `install_aobench`, pas un paquet pip).
- 9 évaluations utilisées : 4 notées automatiquement (`number_prediction`, `mmlu_prediction`, `missing_info`, `sycophancy`) et 5 notées par un juge (`backtracking`, `vagueness`, `domain_confusion`, `activation_sensitivity`, `hallucination`). `taboo` et `personaqa` sont exclues (modèles fine-tunés nécessaires).
- **Juge local Qwen3-32B en bf16** au lieu de Claude Sonnet : pas d'API payante. Nos scores absolus ne sont donc pas comparables au papier ; seules les comparaisons internes comptent.

### Méthodes de sparsité retenues
Critères : sans réentraînement (sinon l'oracle ne lit plus le même modèle), compatibles SwiGLU, écrivables comme hooks PyTorch sur le modèle Hugging Face, appliquées **avant** les couches lues, coupant des **neurones du FFN** (entrée de `down_proj`, unités interprétables), avec un score par neurone et un masque enregistrable.

| Méthode | Score du neurone j | Ce qu'il mesure | Code de référence |
|---|---|---|---|
| **TEAL** (ICLR 2025, [arXiv:2408.14690](https://arxiv.org/abs/2408.14690)) | \|a_j\| | taille de l'activation | [FasterDecoding/TEAL](https://github.com/FasterDecoding/TEAL) (Llama/Mistral, à porter) |
| **WINA** (ICLR 2026, [arXiv:2505.19427](https://arxiv.org/abs/2505.19427)) | \|a_j\| · ‖w_j‖ (w_j = colonne j de W_down) | contribution réelle du neurone à la sortie de la couche | [microsoft/wina](https://github.com/microsoft/wina) (Qwen2.5, proche de Qwen3) |

Réglages décidés :
- **Pas de budget fixé par couche** : pour chaque couche, on prend le plus grand seuil qui garde l'erreur relative de la sortie du FFN sous une tolérance b (2, 5, 10, 20, 30 %), calibré sur quelques milliers de tokens de texte générique (règle empruntée à SATS, [arXiv:2607.08991](https://arxiv.org/abs/2607.08991)). Le seuil étant fixe, le nombre de neurones coupés varie aussi selon le token.
- **WINA sans sa rotation** (inspirée de SliceGPT) : elle change la base et les neurones ne seraient plus ceux d'origine.
- TEAL d'abord, puis WINA. Les deux sont implémentées par **un seul hook** qui reçoit un masque : la méthode ne fait que calculer le score et le masque. Inverser le tri (retirer les plus importants) donne l'**ablation** avec le même code.
- En option pour la sparsité seule : TEAL aussi sur les entrées de q, k, v, o, gate, up (ces masques portent sur le flux résiduel, donc pas utilisables pour l'ablation).
- En réserve : sparsité par attribution gradient × activation (MLSys 2026), SATS, CATS.

### Contrôles à prévoir
- **Masque aléatoire** de même taille, token par token.
- **TEAL uniforme** (même taux partout) pour voir ce que change l'adaptativité.
- **Bruit gaussien de même norme** que l'erreur causée : sépare « le vecteur a bougé » de « une information précise a disparu ».
- Comparer les méthodes **à dégâts égaux** (même KL ou même sparsité moyenne réalisée).
- **Enregistrer la sparsité réalisée** par token, couche et thème : c'est déjà un résultat.

### Stockage des activations : safetensors
Benchmark fait (`benchmark-stockage/`) : même taille et lecture complète que `.pt`, mais lecture d'une seule couche 25× plus rapide qu'un `torch.load` classique, sans pickle.
- Un fichier par thème (catégorie), une clé par couche : `L07.acts` [N, T, 4096] et `L07.mask` [N, T] (padding + masque).
- Lecture : `safe_open(...).get_tensor("L07.acts")`.
- Les activations, données et poids ne vont jamais dans Git (`data/` est local).

### Jeu de données (proposition du 10/11, à finaliser par Raph et Momo)
Tout en anglais (oracle et jeux de référence en anglais). 50 à 100 exemples par thème, avec la bonne réponse connue pour noter automatiquement.

| Thème | Exemple | Source | Rôle |
|---|---|---|---|
| T1 Faits | « The Eiffel Tower is located in the city of » | CounterFact (fait vrai + version fausse) | cœur |
| T2 Contexte | « When Mary and John went to the store, John gave a drink to » | gabarits IOI générés | cœur |
| T3 Calcul | « 23 + 48 = », petits problèmes | script + GSM8K | cœur |
| T4 Texte prévisible / code | « Monday, Tuesday, », `for i in range(10):` | écrit à la main | cœur |
| T5 Sentiment | critique de film | SST-2 | témoin (tâche d'entraînement de l'oracle) |
| T6 Consigne de rôle | « You are a pirate... » | script | témoin |

Les témoins vérifient que l'oracle répond juste en dense, et signalent s'il est perturbé par les activations coupées. Pour RQ1, ajouter des cas **métacognitifs et multi-sauts** (proches des évals AObench `missing_info`, `backtracking`, `sycophancy`).

Questions à l'oracle pour chaque prompt : Q1 description libre (« Can you describe what is happening in this text? »), Q2 prédiction des 5 tokens suivants (automatique), Q3 oui/non propre au thème, Q4 question ouverte propre au thème (mot attendu), Q5 question piège dont la réponse est toujours « non » (mesure l'invention). Q3 et Q5 sur le dernier token, les autres sur plusieurs tokens. Pour les réponses à choix connus, **scorer chaque réponse candidate** (écart de logits Yes − No, AUROC) plutôt que demander une confiance à l'oracle (Torrielli et al., [arXiv:2605.26045](https://arxiv.org/abs/2605.26045)).

---

## 3. Ce qui a été fait

### Infrastructure
- Dépôt structuré, environnement **uv** (Python 3.11, `uv.lock`, torch CUDA 12.6 sur Linux), ruff + pre-commit + nbstripout, tests `pytest`.
- Règles : `uv add` puis commiter `pyproject.toml` et `uv.lock` ensemble, jamais `pip install` (sauf Colab), une branche par tâche.
- Qwen3-8B + oracle tournent sur Colab (`notebooks/tutorial_colab_oracle.ipynb`) : lecture d'une couche et injection dans l'oracle vérifiées.
- **Cluster de l'école (Slurm)** documenté dans `scripts/slurm/README.md` : compte `brain`, partition `Brain_GPU`, `--qos=low`, nœud `sl-mee-br-501` (4 × A100 80 Go), pas de limite de temps mais jobs préemptables (les scripts doivent reprendre où ils se sont arrêtés). Home limité à 20 Go et **illisible par les jobs** : tout vit sur `/Brain/private/ProCom-AO/` (uv, Python, caches HF, repo), chemins fixés par `scripts/slurm/cluster_env.sh`. Télécharger les modèles sur le nœud de login avant (`--prepare`).

### Code existant (`src/procom/`)
- `oracle/aobench.py` (~900 lignes, testé dans `tests/test_aobench.py`) : installation d'AObench, juge local (`LocalJudge`, `use_local_judge`), décalage de la fenêtre de couches lue (`shifted_window`), `ask_oracle`, `run_sweep` / `judge_sweep` avec sauvegarde et reprise, `collect_results`, `aggregate_scores`, `plot_sweep`. Un défaut d'AObench sur la lecture des activations a été corrigé (`target_activations="base"` : activations lues avec le LoRA désactivé ; `prompt_layers="actual"`).
- `scripts/aobench_layer_sweep.py` + `scripts/slurm/aobench_layer_sweep.sbatch` : modèle à suivre pour les prochains scripts (argparse, `--limit`, sauvegarde par unité de travail, reprise).
- `dataset/` et `sparsity/` : **vides** (Raph et Momo). `activations/` : **pas encore créé** (Max et Léo).

### Démonstrateur de sparsité (`notebooks/sparsity_rates_teal_wina.ipynb`, Momo, branche `feature/sparsity-rate-notebook`)
Démonstrateur basique, niveau du tuto (pas une expérience). Retours équipe du 28/11 appliqués : 8 bits autorisé, pas de calibration par tolérance, 2 exemples seulement.
- Qwen3-8B en bf16, ou en **8 bits si GPU < 20 Go** (T4), `down_proj` gardé en bf16 (`llm_int8_skip_modules`) car WINA lit ses poids. La quantification déforme un peu les activations : acceptable pour une démo seulement.
- Scores (`neuron_scores`) : TEAL |a_j|, WINA |a_j|·‖W_down[:, j]‖ (sans rotation).
- Seuils (`calibrate_thresholds`) : les papiers ne donnent pas de valeur de seuil, mais une sparsité cible ; comme TEAL, seuil par couche = quantile 50 % des scores sur un court texte générique (TEAL : 40-50 % de sparsité globale sans grosse perte).
- `run_sparse(messages, method)` : pre-hook sur `mlp.down_proj`, coupe sous le seuil, renvoie `cut[couche, token]`. Affiche sparsité globale, locale par couche, et par token, sur 2 prompts (fait, calcul).
- Vérifié sur CPU avec un Qwen3 aléatoire minuscule ; **version simplifiée pas encore relancée sur Colab** (la version longue du 27/11 y passait).
- Écart assumé avec §2 (seuil par tolérance d'erreur b, SATS) : à faire dans `src/procom/sparsity/` pour les vraies expériences.

### Expérience 1 : où l'oracle sait-il lire ? (terminée, `results/aobench_layer_sweep/`)
19 fenêtres de 5 couches (13-17 à 31-35), 9 évals AObench complètes, juge Qwen3-32B. 1 A100 80 Go : 6 h 21 de génération, 10 h 24 de notation. Décodage glouton, une passe par fenêtre, IC à 95 % par bootstrap.

- Score stable à environ une couche de la fenêtre d'entraînement (20-24 : 0,575 ; 21-25 : 0,570 ; 22-26 : 0,579), puis baisse d'environ 0,01 par couche, et chute nette dès que les couches 33-35 entrent dans la fenêtre (31-35 : 0,398).
- **Fenêtres retenues pour la suite** (aucune couche commune, au plus 10 % de perte par rapport à 21-25) :

| Fenêtre | Score agrégé | Écart vs 21-25 |
|---|---|---|
| 16-20 | 0,516 | −9 % |
| 21-25 (entraînement) | 0,570 | référence |
| 26-30 | 0,532 | −7 % |

- Audit du juge : 237 réponses re-notées à l'aveugle par un second LLM (`judge_audit_items.csv`). Deux évals sont problématiques : `activation_sensitivity` ne discrimine pas (le juge dit presque toujours « différent », et AObench associe 18 de ses 45 paires au mauvais texte) ; `hallucination` : le juge abuse du verdict « obviously wrong », surtout en couches basses (correction dans `hallucination_correction.csv`, une phrase ajoutée à la consigne du juge).
- Les sorties brutes (~114 Mo) ne sont pas dans Git : **lien encore à ajouter** (TODO dans le README des résultats).

Attention : la présentation S9 parlait d'un « plateau 20-32 » issu d'un pré-test à 10 exemples. Le résultat qui fait foi est celui de l'expérience complète ci-dessus. Conséquence pour RQ2 : la fin du réseau (au-delà de la couche 30) est mal lue par l'oracle, il faudra formuler la comparaison « couches médianes vs fin de réseau » avec les fenêtres 16-20 / 21-25 / 26-30, ou la justifier autrement (KL, logit lens).

---

## 4. Points d'attention

- Le mot « oracle » désigne ici l'**Activation Oracle** (un LLM lecteur). Ne pas le confondre avec l'« oracle top-k » (sélecteur parfait des k meilleurs neurones) qui apparaît dans les anciens fichiers `plan/` et `etat-de-l-art/erreur-minimale-sparse.md`.
- Ces anciens fichiers décrivent une **autre piste** (plancher et plafond de l'erreur d'un LLM sparse, sur GPT-2) qui **n'a pas été retenue**. Seule idée réutilisée : le score |a_j| · ‖w_j‖, qui est celui de WINA.
- L'oracle n'a jamais vu d'activations coupées : s'il se trompe, est-ce l'information qui manque ou lui qui est hors distribution ? Parades : témoins T5/T6, contrôles aléatoires et bruit, **AObench rejoué sur activations coupées**.
- Risque principal du planning : la mise en œuvre de la sparsité. Priorité au hook commun et à TEAL.

---

## 5. Prochaines étapes

Priorités immédiates (nov. – déc., S9 à S14) :
1. **Figer le format des fichiers d'activations** (safetensors, un fichier par thème, une clé par couche) et produire des données factices pour que chacun avance.
2. **Module `activations/`** : extraction par hooks des couches lues (fenêtres 16-20, 21-25, 26-30), avec enregistrement de v_dense, v_sparse et Δv.
3. **Module `sparsity/`** : hook commun sur l'entrée de `down_proj`, TEAL puis WINA, calibration des seuils par tolérance d'erreur, masques enregistrables, contrôles (aléatoire, uniforme, bruit). *Hook et scores démontrés dans `notebooks/sparsity_rates_teal_wina.ipynb` (seuils par quantile) : reste à les mettre dans `src/procom/sparsity/` (+ tests) avec la calibration par tolérance, les masques enregistrés et les contrôles.*
4. **Métriques de sortie** : perplexité, KL dense / sparse, exactitude, sparsité réalisée.
5. **Module `dataset/`** : générer les 4 thèmes + 2 témoins (+ cas métacognitifs / multi-sauts) et les questions Q1-Q5.
6. **AObench sur activations coupées**, sur les trois fenêtres retenues, pour mesurer la fiabilité de l'oracle hors distribution.
7. Ajouter le lien des sorties brutes de l'expérience 1.

Ensuite : passes dense / sparse sur tout le jeu (janvier, revue 2 en S16), puis analyses RQ1, RQ2, RQ3, démonstrateur et article (février – mars, revue 3 en S25).

---

## 6. Journal (une ligne par push, la plus récente en haut)

- 28/11/2026 · `feature/sparsity-rate-notebook` · notebook simplifié en démonstrateur (8 bits si besoin, seuils = quantile 50 % façon TEAL, 2 exemples ; plus de calibration par tolérance, figures ni sauvegarde).
- 27/11/2026 · `feature/sparsity-rate-notebook` · ajout `notebooks/sparsity_rates_teal_wina.ipynb` (hook `down_proj`, calibration TEAL/WINA par tolérance, sparsité locale/globale) ; `CONTEXTE.md` versionné, carte §0 + journal §6.
