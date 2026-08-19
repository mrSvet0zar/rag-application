# Roadmap

Feuille de route pour faire passer le projet d'un MVP fonctionnel à un système
qu'on assumerait en production. Chaque chantier est indépendant ; l'ordre
recommandé est en fin de document.

---

## ✅ Chantier 0 — Fondations (fait)

- Architecture en injection de dépendances, interfaces `Protocol` (`Embedder`,
  `Reranker`, `VectorStore`, `Generator`) — les pièces du pipeline sont
  substituables.
- Erreurs de domaine (`AppError`) traduites en HTTP à un seul endroit ; les
  services ne dépendent pas de FastAPI.
- ~70 tests : unitaires (doubles déterministes) + intégration sur un **vrai**
  pgvector, plus tests d'API bout-en-bout.
- CI : ruff (lint + format), mypy (`disallow_untyped_defs`), pytest + seuil de
  couverture, eslint/prettier, build de l'image Docker.
- Reranking cross-encoder, ingestion multi-formats (txt/md/pdf/docx/html + URL
  avec garde SSRF), streaming SSE, thème clair/sombre persistant.

---

## Chantier A — Qualité du RAG : mesurer, puis optimiser

**L'idée directrice : on ne peut pas optimiser ce qu'on ne mesure pas.**
L'évaluation vient donc *avant* les optimisations, pour que chacune devienne une
expérience chiffrée plutôt qu'une intuition.

### A1. Harness d'évaluation + jeu de référence ✅

- **Golden set** versionné (`backend/eval/golden_set.json`) : ~30-50 questions
  annotées avec les chunks attendus. C'est ~40 % de l'effort du chantier, et la
  fondation de tout le reste.
- **Métriques de retrieval déterministes** : `recall@k`, `precision@k`, `MRR`,
  `nDCG@k`. Rapides, gratuites, sans LLM → **exécutables en CI** comme garde-fou
  anti-régression.
- **Métriques de génération (LLM-as-judge)** : *faithfulness* (la réponse est-elle
  fondée sur le contexte ?), *answer relevance*. Coûteuses et non déterministes →
  **hors CI**, lancées à la demande.
- Mesurer aussi **latence (p50/p95) et coût en tokens** : la qualité seule ne dit
  rien du compromis réel.

> ⚠️ Ne jamais mettre les métriques LLM-as-judge en gate CI : coût par push,
> résultats instables, dépendance réseau.

### A2. Baseline ✅

**Faite.** Résultats et méthode dans [EVALUATION.md](EVALUATION.md).
Vectoriel seul : hit@5 0.419, MRR 0.309, p50 22 ms. Avec reranking :
hit@5 0.548, MRR 0.492, p50 560 ms — soit +31 % de hit@5 et +59 % de MRR
pour ×25 de latence.

Deux constats orientent la suite : l'écart entre `hit@5` (0.55) et
`doc_hit@5` (0.77) montre que le bon article est souvent trouvé sans le bon
passage — c'est un problème de découpage (A4) ; et les questions à terme
exact restent mal servies (le titre *Attention Is All You Need* sort au 37ᵉ
rang), ce qui chiffre l'intérêt de l'hybride (A3).

### A3. Recherche hybride (lexical + vectoriel) ✅

**Faite.** Colonne `tsvector` générée + index GIN (migration `0002_lexical`),
recherche full-text dans `Database.search_lexical`, fusion **RRF** dans
`app/fusion.py`, câblée dans `Retriever`. Réglable par `HYBRID_ENABLED`.

Résultats dans [EVALUATION.md](EVALUATION.md) : `hit@5` 0.419 → **0.774**,
MRR 0.309 → **0.664**. Les 12 questions à terme exact passent toutes.

Trois choses apprises en la construisant :
- `websearch_to_tsquery` combine les termes en **ET** → zéro résultat sur une
  question en langage naturel ; il faut combiner les lexèmes en **OU**.
- Le budget du cross-encoder doit être **distinct** du pool par moteur, sinon la
  troncature jette l'apport lexical avant de le juger (`RERANK_MAX_CANDIDATES`).
- Une question du golden set était fausse (réponse présente uniquement en
  bibliographie) : corrigée, incident documenté.

### A4. A/B testing des stratégies de chunking ✅

**Fait.** `python -m eval.runner --sweep` compare 256/512/1024/2048, à retrieval
constant, dans deux vues : à k constant et à budget de contexte constant (cinq
chunks de 256 caractères ne donnent pas au LLM le même texte que cinq de 1024).

**512/100 retenu** et adopté comme défaut : meilleur `hit@k` et `doc_hit@k` dans
les deux vues, et **moitié moins de latence** que 1024/200. Résultats détaillés
dans [EVALUATION.md](EVALUATION.md).

Enseignement principal : le découpage **interagit** avec le reranking. Passer à
512 dégrade l'hybride seul (0.645 → 0.484) mais améliore l'ensemble avec
reranking (0.774 → 0.806). Choisir un découpage sans tenir compte de l'étage
suivant aurait mené à la mauvaise conclusion.

### A5. Livrable ✅

Tableaux de résultats publiés dans [EVALUATION.md](EVALUATION.md) et résumés
dans le README, avec la méthode, les réglages découverts en mesurant, l'incident
de golden set et les limites assumées.

---

## Chantier B — Production-readiness

Constats issus de l'audit du code actuel, par ordre de gravité.

### B1. Migrations de schéma (Alembic) ✅

**Fait.** Alembic est désormais la source unique du schéma ; `init_db.sql` a
été supprimé. Le backend applique `alembic upgrade head` au démarrage, sous
verrou consultatif PostgreSQL pour que deux instances ne migrent pas
simultanément. La révision de référence est idempotente, donc la base de
production existante est adoptée sans perte de données (vérifié : 3 documents,
6 chunks, 27 conversations, 52 messages intacts après estampillage).
Quatre tests d'intégration couvrent le contrat de migration, dont l'adoption
d'une base portant déjà l'ancien schéma.

### B2. Taille d'upload non bornée ✅

**Fait.** Limite unique `MAX_UPLOAD_BYTES` (10 Mo par défaut) appliquée aux deux
chemins d'entrée :

* Uploads : middleware ASGI, car FastAPI parse le multipart *avant* d'entrer
  dans la fonction — un contrôle dans l'endpoint arriverait trop tard. Deux
  couches : rejet immédiat sur `Content-Length`, et comptage en flux pour les
  requêtes chunked qui n'en déclarent pas.
* Import d'URL : le contrôle existant était cosmétique (`resp.content` avait
  déjà tout chargé en mémoire). Le téléchargement est maintenant streamé et
  interrompu dès le dépassement (`read_capped`).

Cinq tests d'intégration + quatre unitaires ; comportement confirmé aussi sous
uvicorn réel (11 Mo → 413, upload normal → 200).

### B3. Ingestion synchrone dans la requête 🟠

Extraction + chunking + embeddings se font dans le cycle de la requête HTTP. Un
gros PDF = timeout, aucune reprise, aucune progression côté client. La colonne
`documents.status` (`processing`/`completed`/`failed`) existe déjà mais n'est
jamais exploitée de façon asynchrone.
→ File de jobs (ARQ/Celery) ou, a minima, tâche de fond + polling du statut.

### B4. Pas de rate limiting 🟠

Endpoints publics déclenchant des appels LLM facturés : n'importe qui peut vider
le crédit Anthropic. → Limitation par IP (slowapi/Redis) sur `/chat*` et l'ingestion.

### B5. Healthcheck superficiel 🟠

`/health` renvoie `ok` sans vérifier la base : le healthcheck Railway reste vert
alors que l'application est incapable de répondre.
→ Séparer *liveness* (`/health`) et *readiness* (`/ready`, avec ping DB).

### B6. Observabilité 🟠

Logs non structurés, pas d'identifiant de corrélation, aucune métrique.
Impossible de relier un incident utilisateur à une trace.
→ Logs JSON + `request_id` (middleware), métriques de latence par étape
(embed / search / rerank / LLM) et coût en tokens, Sentry.

### B7. Durcissement de l'image 🟢

Le conteneur tourne en **root** et ne déclare pas de `HEALTHCHECK`.
→ Utilisateur non privilégié, `HEALTHCHECK`, build multi-stage.

### B8. Détails d'API 🟢

- Pas de pagination sur `GET /documents` ni de borne sur l'historique de
  conversation.
- CORS : `allow_methods=["*"]` combiné à `allow_credentials=True`.
- Pas de versionnement d'API (`/api/v1`).

### B9. Tests frontend 🟢

Le frontend n'a que du lint/format — **aucun test**. Le parsing SSE et le rendu
markdown maison sont les zones les plus fragiles.
→ Vitest + Testing Library sur le client SSE, le hook de thème, le rendu des sources.

---

## Ordre recommandé

1. ~~**B1** (migrations)~~ ✅
2. ~~**B2** (limite d'upload)~~ ✅
3. ~~**A1 → A2** (harness + baseline)~~ ✅
4. ~~**A3 → A4 → A5**~~ ✅ (hybride, A/B chunking, tableau de résultats).
5. **B3 → B6** (asynchrone, rate limit, readiness, observabilité).
6. **B7 → B9** (durcissement, API, tests frontend).
