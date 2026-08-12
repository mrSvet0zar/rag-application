# Notes sur pgvector

`pgvector` est une extension PostgreSQL open-source qui ajoute un type de donnée
`vector` ainsi que des opérateurs de distance pour la recherche de similarité.

## Opérateurs de distance

- `<->` : distance euclidienne (L2)
- `<=>` : distance cosinus
- `<#>` : produit scalaire négatif (inner product)

Pour obtenir une **similarité** cosinus dans l'intervalle [-1, 1] à partir de la
distance cosinus, on calcule `1 - (a <=> b)`.

## Index de recherche approximative

Pour accélérer la recherche sur de grands volumes, pgvector propose deux types
d'index :

- **IVFFlat** : partitionne l'espace en listes (clusters). Rapide à construire,
  bon compromis. Il faut choisir le paramètre `lists` (souvent ~ sqrt(nombre de
  lignes)).
- **HNSW** : graphe de navigation hiérarchique. Plus lent à construire mais
  offre un meilleur rappel à latence égale.

Dans ce projet, nous utilisons un index **HNSW** avec l'opérateur
`vector_cosine_ops`, adapté à des embeddings normalisés. HNSW est préféré à
IVFFlat car il se construit de façon incrémentale et reste fiable même avec peu
de données, alors qu'IVFFlat doit être entraîné après insertion et peut ne rien
renvoyer sur un petit volume s'il est mal paramétré.

## Dimension des vecteurs

La dimension du vecteur doit correspondre exactement à la sortie du modèle
d'embeddings. Le modèle multilingue MiniLM utilisé ici produit des vecteurs de
**384 dimensions**, d'où la colonne `embedding vector(384)`.
