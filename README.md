# 🤖 RAG Application — Chatbot sur documents personnalisés

Chatbot de **Retrieval-Augmented Generation** : uploadez vos documents, posez
des questions, obtenez des réponses **ancrées dans vos sources** (avec citation
des passages utilisés et scores de similarité).

Ce projet fait partie d'un portfolio démontrant des compétences autour de l'IA,
des LLMs et de la recherche sémantique.

---

## 🧱 Stack

| Couche       | Techno                                                                 |
| ------------ | ---------------------------------------------------------------------- |
| Frontend     | React 18 + Vite + Tailwind CSS + React Query                           |
| Backend      | FastAPI (async) + asyncpg                                              |
| Embeddings   | `sentence-transformers` (local, multilingue, **384 dims**, sans clé)   |
| LLM          | Anthropic **Claude** (`claude-sonnet-5` par défaut)                    |
| Chunking     | LangChain `RecursiveCharacterTextSplitter`                             |
| Reranking    | Cross-encoder multilingue (`mmarco-mMiniLMv2`) — re-note les candidats |
| Vector store | PostgreSQL 16 + **pgvector** (index HNSW, similarité cosinus)          |

> **Note embeddings** : Anthropic ne propose pas d'API d'embeddings. On utilise
> donc un modèle local (gratuit, aucune clé). Claude n'intervient que pour la
> **génération** de la réponse finale.

> **Mode démo** : sans clé Anthropic, l'app reste 100 % fonctionnelle — la
> recherche vectorielle marche et la réponse est construite localement à partir
> des passages récupérés. Ajoutez `ANTHROPIC_API_KEY` pour activer Claude.

---

## 🏗️ Architecture

```
Frontend (React)  ──HTTP──►  Backend (FastAPI)
                                 │
                 ┌───────────────┼────────────────┐
                 ▼               ▼                ▼
         Embeddings local   RAG pipeline     PostgreSQL + pgvector
        (sentence-transf.)  (chunk + Claude)  (documents, chunks,
                                               conversations, messages)
```

**Flux d'une question :** embedding de la question → recherche d'un pool de
candidats (cosinus) → **reranking cross-encoder** (re-note finement puis garde
les meilleurs) → injection dans le prompt → génération Claude → persistance de
la conversation → réponse + sources (avec score de reranking).

---

## 🚀 Démarrage rapide

### Prérequis
- Docker (pour PostgreSQL + pgvector)
- Python 3.10+
- Node.js 18+

### 1. Base de données

```bash
docker compose up -d
```

Cela démarre PostgreSQL sur le port `5432` et applique automatiquement
`backend/init_db.sql` (extension pgvector + tables + index).

### 2. Backend

```bash
cd backend
python -m venv venv
# Windows PowerShell:  venv\Scripts\Activate.ps1
# macOS/Linux:         source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # puis, optionnel : renseignez ANTHROPIC_API_KEY
uvicorn app.main:app --reload --port 8000
```

- API : http://localhost:8000
- Docs interactives (Swagger) : http://localhost:8000/docs

> Au premier appel nécessitant un embedding, le modèle (~120 Mo) est téléchargé
> puis mis en cache.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Interface : http://localhost:5173

### 4. (Optionnel) Charger les documents d'exemple

Trois documents markdown sont fournis dans `backend/sample_docs/`.
Serveur démarré, dans un autre terminal :

```bash
cd backend
python -m scripts.seed_documents
```

…ou glissez-les simplement dans l'interface.

---

## 📡 API

| Méthode  | Route                              | Description                          |
| -------- | ---------------------------------- | ------------------------------------ |
| `GET`    | `/api/health`                      | Statut + mode démo                   |
| `POST`   | `/api/documents/upload`            | Upload (txt/md/pdf/docx/html) + index |
| `POST`   | `/api/documents/import-url`        | Importe une page web (SSRF-guarded)  |
| `GET`    | `/api/documents`                   | Liste des documents                  |
| `DELETE` | `/api/documents/{id}`              | Supprime un document et ses chunks   |
| `POST`   | `/api/chat`                        | Pose une question (RAG, réponse complète) |
| `POST`   | `/api/chat/stream`                 | Idem en **streaming** (SSE, token par token) |
| `GET`    | `/api/conversations/{id}`          | Historique d'une conversation        |
| `GET`    | `/api/stats`                       | Statistiques globales                |

Exemple de requête chat :

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Qu'\''est-ce que le RAG ?", "k": 5}'
```

---

## 🧪 Tests

```bash
cd backend
pytest          # tests unitaires (chunking, prompt, mode démo — sans DB ni modèle)
```

---

## ⚙️ Configuration (backend/.env)

| Variable              | Défaut                          | Rôle                                        |
| --------------------- | ------------------------------- | ------------------------------------------- |
| `ANTHROPIC_API_KEY`   | *(vide)*                        | Clé Claude. Vide = mode démo.               |
| `CHAT_MODEL`          | `claude-sonnet-5`               | Modèle de génération                        |
| `EMBEDDING_MODEL`     | `…paraphrase-multilingual-MiniLM-L12-v2` | Modèle d'embeddings local          |
| `PGVECTOR_DIMENSION`  | `384`                           | Dimension des vecteurs (⇔ modèle)           |
| `CHUNK_SIZE`          | `1024`                          | Taille des chunks (caractères)              |
| `CHUNK_OVERLAP`       | `200`                           | Chevauchement entre chunks                  |
| `MIN_RELEVANCE_SCORE` | `0.25`                          | Seuil de similarité minimal (sans rerank)   |
| `TOP_K_RETRIEVAL`     | `5`                             | Nombre de chunks récupérés                  |
| `RERANK_ENABLED`      | `true`                          | Active le reranking cross-encoder (2ᵉ modèle, ~500 Mo RAM) |
| `RERANK_MODEL`        | `…mmarco-mMiniLMv2-L12-H384-v1` | Modèle cross-encoder multilingue            |
| `RERANK_CANDIDATES`   | `20`                            | Taille du pool re-noté par le cross-encoder |
| `RERANK_MIN_SCORE`    | `0.05`                          | Seuil du score de rerank (garde ≥ 1 source) |

> ⚠️ Si vous changez `EMBEDDING_MODEL`, mettez à jour `PGVECTOR_DIMENSION` **et**
> la dimension `vector(...)` dans `init_db.sql`, puis recréez la base.

---

## 📁 Structure

```
rag-application/
├── docker-compose.yml         # PostgreSQL + pgvector
├── backend/
│   ├── init_db.sql            # schéma + extension + index
│   ├── requirements.txt
│   ├── .env.example
│   ├── app/
│   │   ├── config.py          # settings (pydantic-settings)
│   │   ├── schemas.py         # modèles Pydantic (API)
│   │   ├── embeddings.py      # embeddings locaux (sentence-transformers)
│   │   ├── rag_pipeline.py    # chunking + génération Claude / démo
│   │   ├── vector_db.py       # accès DB async + recherche vectorielle
│   │   └── main.py            # app FastAPI + routes
│   ├── scripts/seed_documents.py
│   ├── sample_docs/           # documents d'exemple
│   └── tests/
└── frontend/
    └── src/
        ├── App.jsx
        ├── services/api.js
        └── components/{Navbar,DocumentPanel,ChatInterface}.jsx
```

---

## ☁️ Déploiement

Guide complet Railway (backend + DB) + Vercel (frontend) :
**[DEPLOYMENT.md](DEPLOYMENT.md)**. Fichiers fournis : `backend/Dockerfile`,
`backend/railway.toml`, `frontend/vercel.json`, `.github/workflows/ci.yml`.

---

## 🗺️ Améliorations possibles (post-MVP)

- ✅ ~~Réponses en streaming (SSE)~~ — fait
- ✅ ~~Reranking avec un cross-encoder~~ — fait (`RERANK_ENABLED`)
- ✅ ~~Support de plus de formats (docx, html, import d'URL)~~ — fait
- Authentification utilisateur
- Support de plus de formats (docx, html, crawling web)
- Déploiement (Railway + Vercel)
