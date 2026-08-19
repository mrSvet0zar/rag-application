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

### A3. Recherche hybride (lexical + vectoriel)

La recherche vectorielle seule échoue sur les termes exacts : références,
acronymes, noms propres, jargon rare. Le lexical couvre exactement ce trou.

- **Pas d'Elasticsearch** : PostgreSQL fait du full-text nativement
  (`tsvector` + index GIN + `ts_rank_cd`), fusionnable avec pgvector dans **une
  seule requête**.
- Fusion par **RRF** (Reciprocal Rank Fusion) plutôt que somme pondérée : évite
  d'avoir à normaliser deux échelles de scores incomparables.
- Corpus francophone → configuration `french` + `unaccent`, pas le défaut anglais.
- **Honnêteté de vocabulaire** : `ts_rank_cd` n'est pas BM25 au sens strict. On
  parle de « recherche lexicale full-text », ou on passe par ParadeDB/`pg_search`
  pour du vrai BM25 (au prix d'un déploiement plus lourd).
- Nécessite une colonne `tsvector` sur `chunks` → **dépend de B1 (migrations)**.

### A4. A/B testing des stratégies de chunking

Quasi gratuit une fois A1 en place : on relance le harness en faisant varier
`chunk_size`, `chunk_overlap`, les séparateurs. Extensible aux poids de fusion,
au seuil de rerank et à `top_k`.

### A5. Livrable

Tableau de résultats dans le README — c'est l'artefact qui prouve la démarche :

| Configuration | recall@5 | nDCG@5 | faithfulness | p95 |
|---|---|---|---|---|
| Vectoriel seul | | | | |
| Vectoriel + rerank | | | | |
| Hybride + rerank | | | | |

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
4. **A3 → A4 → A5** (hybride, A/B chunking, tableau de résultats).
5. **B3 → B6** (asynchrone, rate limit, readiness, observabilité).
6. **B7 → B9** (durcissement, API, tests frontend).
