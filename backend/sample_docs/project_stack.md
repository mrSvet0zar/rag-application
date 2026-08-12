# Stack technique du projet RAG

Ce projet démontre la maîtrise d'une chaîne RAG complète, de bout en bout.

## Backend

- **FastAPI** sert l'API REST asynchrone.
- **asyncpg** gère l'accès à PostgreSQL de façon non bloquante.
- **sentence-transformers** produit les embeddings localement, sans clé API et
  sans coût, avec un modèle multilingue (français et anglais).
- **Anthropic Claude** génère la réponse finale à partir du contexte récupéré.
- **LangChain text-splitters** découpe les documents en chunks.

## Base de données

PostgreSQL 16 avec l'extension **pgvector** stocke à la fois les métadonnées des
documents et les embeddings des chunks. Quatre tables principales : `documents`,
`chunks`, `conversations` et `messages`.

## Frontend

Une interface **React + Vite** stylée avec **Tailwind CSS** permet d'uploader
des documents, de poser des questions et de visualiser les sources citées pour
chaque réponse.

## Mode démo

Si aucune clé Anthropic n'est fournie, l'application fonctionne quand même : la
recherche vectorielle reste active et une réponse est construite localement à
partir des passages récupérés. Cela permet de tester toute la chaîne sans
dépendre d'un service payant.
