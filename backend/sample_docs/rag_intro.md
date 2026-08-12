# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) est une architecture qui combine un modèle
de langage (LLM) avec une base de connaissances externe. Au lieu de s'appuyer
uniquement sur les paramètres appris pendant l'entraînement, le modèle va
d'abord **récupérer** les passages les plus pertinents dans un corpus de
documents, puis les utiliser comme contexte pour **générer** sa réponse.

## Pourquoi utiliser le RAG ?

- **Réduction des hallucinations** : les réponses sont ancrées dans des sources
  réelles et vérifiables.
- **Fraîcheur de l'information** : on peut mettre à jour la base documentaire
  sans réentraîner le modèle.
- **Traçabilité** : chaque réponse peut citer les documents sources utilisés.
- **Confidentialité** : les documents privés restent dans votre infrastructure.

## Les étapes d'un pipeline RAG

1. **Ingestion** : les documents sont découpés en morceaux (chunks) de taille
   raisonnable, avec un léger chevauchement pour préserver le contexte.
2. **Embedding** : chaque chunk est transformé en vecteur numérique par un
   modèle d'embeddings.
3. **Indexation** : les vecteurs sont stockés dans une base vectorielle
   (ici PostgreSQL avec l'extension pgvector).
4. **Recherche** : à la question de l'utilisateur, on calcule l'embedding de la
   question et on récupère les chunks les plus proches par similarité cosinus.
5. **Génération** : les chunks récupérés sont injectés dans le prompt du LLM,
   qui rédige une réponse fondée sur ce contexte.

## Similarité cosinus

La similarité cosinus mesure l'angle entre deux vecteurs. Une valeur proche de 1
signifie que les textes sont sémantiquement très proches, tandis qu'une valeur
proche de 0 indique qu'ils n'ont pas de rapport. C'est la métrique de choix pour
comparer des embeddings normalisés.
