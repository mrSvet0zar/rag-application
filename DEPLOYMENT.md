# 🚀 Guide de déploiement — Railway (backend + DB) + Vercel (frontend)

Ce guide déploie l'application en production :

- **PostgreSQL + pgvector** → Railway
- **Backend FastAPI** → Railway (image Docker)
- **Frontend React** → Vercel

> ⚠️ **Mémoire** : le backend charge un modèle d'embeddings local (torch +
> sentence-transformers), ce qui demande **~1 Go de RAM**. Le palier gratuit de
> Railway peut être juste ; prévoyez le plan « Hobby » si le service redémarre
> pour cause d'OOM. (Alternative plus légère : passer à un embedding ONNX via
> `fastembed`, ou à une API d'embeddings — voir la fin du guide.)

---

## 0. Prérequis

- Le projet poussé sur un dépôt GitHub.
- Un compte [Railway](https://railway.app) et un compte [Vercel](https://vercel.com).
- Une clé `ANTHROPIC_API_KEY` (sinon le backend tourne en mode démo).

---

## 1. Base de données PostgreSQL + pgvector (Railway)

Le PostgreSQL par défaut de Railway n'embarque pas toujours `pgvector`. Le plus
simple est de déployer **l'image officielle pgvector** comme service :

1. Dans votre projet Railway : **New → Empty Service** (ou *Deploy a Docker Image*).
2. Image : `pgvector/pgvector:pg16`.
3. Variables d'environnement du service DB :
   ```
   POSTGRES_USER=rag_user
   POSTGRES_PASSWORD=<mot-de-passe-fort>
   POSTGRES_DB=rag_db
   ```
4. Ajoutez un **volume** monté sur `/var/lib/postgresql/data` (persistance).

> ✅ **Pas de SQL à exécuter à la main** : au premier démarrage, le backend
> applique automatiquement `init_db.sql` (extension pgvector + tables + index
> HNSW). C'est idempotent — il vérifie à chaque démarrage sans rien casser.
> Il vous suffit donc de fournir la base ; le schéma se crée tout seul.

> 💡 Alternatives managées avec pgvector inclus : **Neon** ou **Supabase**
> (créez la base, récupérez l'URL de connexion — le schéma s'appliquera au
> démarrage du backend, comme ci-dessus).

Notez l'URL de connexion interne, de la forme :
`postgresql://rag_user:<pwd>@<host>:5432/rag_db`

---

## 2. Backend FastAPI (Railway)

1. **New → GitHub Repo**, sélectionnez votre dépôt.
2. Dans les **Settings** du service :
   - **Root Directory** : `backend`
     (Railway lira alors `backend/Dockerfile` et `backend/railway.toml`.)
3. **Variables d'environnement** :
   ```
   DATABASE_URL=postgresql://rag_user:<pwd>@<db-host>:5432/rag_db
   ANTHROPIC_API_KEY=sk-ant-...        # vide = mode démo
   CHAT_MODEL=claude-sonnet-5
   EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   PGVECTOR_DIMENSION=384
   MIN_RELEVANCE_SCORE=0.25
   TOP_K_RETRIEVAL=5
   CHUNK_SIZE=1024
   CHUNK_OVERLAP=200
   CORS_ORIGINS=https://<votre-app>.vercel.app
   ```
   > `PORT` est injecté automatiquement par Railway — ne pas le définir.
   > `CORS_ORIGINS` doit contenir l'URL exacte du frontend Vercel (voir étape 3).
4. Déployez. Le healthcheck cible `/api/health`.
   Le build pré-télécharge le modèle : la **première build est longue** (~5–10 min).
5. Récupérez l'URL publique du backend : `https://<backend>.up.railway.app`.

---

## 3. Frontend React (Vercel)

1. **Add New → Project**, importez le dépôt GitHub.
2. **Root Directory** : `frontend`.
   Vercel détecte Vite ; `frontend/vercel.json` fournit build + rewrite SPA.
3. **Environment Variables** :
   ```
   VITE_API_URL=https://<backend>.up.railway.app
   VITE_API_TIMEOUT=60000
   ```
4. Déployez. Récupérez l'URL : `https://<votre-app>.vercel.app`.

---

## 4. Boucler la configuration CORS

Une fois l'URL Vercel connue, revenez sur Railway (backend) et assurez-vous que
`CORS_ORIGINS` la contient **exactement** (sans slash final), puis redéployez :

```
CORS_ORIGINS=https://<votre-app>.vercel.app
```

Vous pouvez en mettre plusieurs, séparées par des virgules.

---

## 5. Vérification post-déploiement

```bash
# Backend en ligne + mode
curl https://<backend>.up.railway.app/api/health

# Stats
curl https://<backend>.up.railway.app/api/stats
```

Puis, dans le frontend Vercel : uploadez un document, posez une question,
vérifiez que la réponse se streame et cite ses sources.

Pour charger les documents d'exemple :
```bash
VITE_API_URL=https://<backend>.up.railway.app python -m scripts.seed_documents
```

---

## 🔁 CI/CD

Railway et Vercel se redéploient **automatiquement** à chaque `git push` sur la
branche connectée. Rien de plus à configurer pour un CD basique.

Pour un pipeline GitHub Actions (lint + tests avant déploiement), un point de
départ :

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r backend/requirements.txt
      - run: cd backend && pytest -q
```

---

## 🪶 Réduire l'empreinte mémoire (optionnel)

Si le backend dépasse la RAM disponible :

- **Option A — embeddings ONNX** : remplacer `sentence-transformers`/`torch` par
  [`fastembed`](https://github.com/qdrant/fastembed) (ONNX, pas de torch,
  empreinte bien moindre). Choisir un modèle 384 dims pour garder le schéma.
- **Option B — embeddings via API** : utiliser une API d'embeddings (p. ex.
  OpenAI `text-embedding-3-small`, 1536 dims → adapter `PGVECTOR_DIMENSION` et
  la colonne `vector(...)`).
- **Option C** : monter le plan Railway (Hobby) pour disposer de plus de RAM.

Seul le module `app/embeddings.py` est à modifier pour les options A/B — le reste
du pipeline est inchangé.
