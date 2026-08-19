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

### B3. Ingestion asynchrone ✅

**Faite.** L'upload et l'import d'URL répondent **202 Accepted** avec la ligne en
`processing` ; le découpage et les embeddings se font en tâche de fond, et le
client sonde `GET /api/documents/{id}`. Mesuré : un document de 445 chunks
répond en **0,04 s** au lieu de bloquer **38 s**.

Ce qu'il a fallu traiter, et qui n'existait pas en synchrone :
- **Réconciliation au démarrage** : une ingestion interrompue par un redémarrage
  ne peut pas reprendre ; sa ligne resterait en `processing` indéfiniment. Elle
  est passée à `failed` au boot — un échec honnête plutôt qu'un zombie.
- **Concurrence bornée** (sémaphore) : chaque job garde un document en mémoire
  et sature un cœur ; des jobs illimités épuiseraient le conteneur bien avant
  que le rate limiter ne s'en aperçoive.
- **`process` ne lève jamais** : il n'y a plus personne à qui remonter
  l'exception, donc l'issue est écrite sur la ligne, que le client sonde.
- Les validations bon marché (fichier illisible, aucun texte, aucun chunk)
  restent **dans la requête**, tant qu'il y a quelqu'un pour les entendre.

> Limite assumée : les tâches de fond vivent dans le processus. C'est adapté à
> une instance unique, et la réconciliation couvre le redémarrage. Passer à une
> vraie file (ARQ/Celery + Redis) deviendrait justifié avec plusieurs répliques,
> des reprises automatiques ou des ingestions de plusieurs minutes.

### B4. Rate limiting ✅

**Fait.** Seau à jetons par client, appliqué aux **seuls** endpoints coûteux
(chat, upload, import d'URL) : une question déclenche un appel LLM facturé, une
lecture non. Un seau plutôt qu'une fenêtre fixe, qui laisserait dépenser un
quota complet à la fin d'une fenêtre puis à nouveau au début de la suivante.

L'identification du client est le point délicat : derrière un proxy,
`scope["client"]` est le proxy (tout le monde partagerait un seau) tandis que
`X-Forwarded-For` est forgeable si rien de confiance ne le réécrit — d'où
`TRUST_PROXY_HEADERS`, qui en fait une décision de déploiement explicite.

> Limite assumée : les seaux sont en mémoire du processus. Avec plusieurs
> répliques, chacune applique la limite séparément.

### B5. Liveness et readiness séparées ✅

**Fait.** `/api/health` ne fait aucune I/O (une base lente ne doit pas faire
tuer un processus sain) ; `/api/ready` fait un aller-retour SQL et répond
**503** sinon, pour qu'on retire l'instance du trafic au lieu d'échouer devant
l'utilisateur. Le healthcheck Railway et celui du Dockerfile pointent dessus.

### B6. Observabilité ✅

**Fait.** Logs **JSON** (tout ce qui est passé en `extra` devient une clé de
premier niveau, donc requêtable), identifiant de requête repris de
`X-Request-ID` s'il existe et renvoyé dans la réponse, porté par un `contextvar`
qui suit les `await`. Le middleware est ASGI et le plus externe : les 404, les
erreurs de validation et les 413 sont donc tracés eux aussi — précisément les
réponses dont on vous parle.

Découpage des latences **par étage** sur les deux endpoints de chat
(récupération / génération / persistance, plus le *time to first token* en
streaming). « Deux secondes » n'est pas actionnable ; « dont 1,8 dans le
cross-encoder » l'est.

Détail attrapé en vérifiant : `env.py` d'Alembic appelait `fileConfig()`, ce qui
réinitialisait le logging du processus et remplaçait silencieusement le
formateur JSON par celui d'`alembic.ini`. Isolé au seul usage CLI.

### B7. Durcissement de l'image ✅

**Fait.** Le conteneur tourne sous un utilisateur non privilégié (uid 10001) et
déclare un `HEALTHCHECK` pointant sur `/api/ready`, avec un `start-period`
couvrant les migrations et le chargement des modèles. Vérifié : image
construite, conteneur `healthy` en 5 s, `id` retourne bien `appuser`.

> Le multi-stage a été **écarté sciemment** : l'essentiel de l'image est PyTorch
> et deux modèles pré-téléchargés, qui doivent de toute façon s'y trouver, et
> les wheels n'exigent aucun compilateur. La complexité n'achèterait presque
> rien.

### B8. Détails d'API ✅

**Fait.** Pagination bornée et plafonnée sur `GET /api/documents`
(`limit` ≤ 200), historique de conversation limité aux messages les plus
**récents** (et non aux plus anciens — sinon on afficherait le début d'une
longue conversation), et CORS resserré : méthodes et en-têtes énumérés plutôt
que `*`, ce qui est à la fois plus large que nécessaire et refusé par certains
navigateurs quand les credentials sont autorisés.

> **Versionnement d'URL (`/api/v1`) : considéré et écarté.** Il n'a de valeur
> qu'avec des consommateurs externes à ne pas casser ; ici le frontend est le
> seul client et se déploie avec l'API. L'introduire imposerait un déploiement
> coordonné (Railway et Vercel déploient indépendamment, donc une fenêtre de
> désaccord) pour un bénéfice nul aujourd'hui. À faire le jour où un tiers
> consomme l'API.

### B9. Tests frontend ✅

**Fait.** Vitest + Testing Library, 24 tests, exécutés en CI. Ciblés sur ce qui
est réellement fragile plutôt que sur du rendu trivial :

- **Le client SSE**, écrit à la main : une frontière de lecture réseau peut
  tomber au milieu d'une trame, au milieu de plusieurs trames, ou **au milieu
  d'un caractère UTF-8** — les trois sont couverts.
- **L'échappement**, qui est ici une propriété de sécurité : le texte de
  l'assistant passe par `dangerouslySetInnerHTML`. Un `<img onerror>` dans une
  réponse ou dans un extrait de source doit rester du texte.
- **Le thème** : préférence système, choix stocké prioritaire, valeur stockée
  aberrante ignorée.
- **Les badges de source** : rerank, repli sur le cosinus, et libellé « lexical »
  quand aucun score de similarité n'existe — plutôt qu'un 0 % inventé.

## Ordre recommandé

1. ~~**B1** (migrations)~~ ✅
2. ~~**B2** (limite d'upload)~~ ✅
3. ~~**A1 → A2** (harness + baseline)~~ ✅
4. ~~**A3 → A4 → A5**~~ ✅ (hybride, A/B chunking, tableau de résultats).
5. ~~**B3 → B6**~~ ✅ (asynchrone, rate limit, readiness, observabilité).
6. ~~**B7 → B9**~~ ✅ (durcissement, API, tests frontend).
