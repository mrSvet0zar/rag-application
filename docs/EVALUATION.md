# Évaluation de la recherche

On ne peut pas optimiser ce qu'on ne mesure pas. Ce document décrit comment la
qualité de la récupération est mesurée dans ce projet, et donne les résultats de
référence auxquels toute évolution ultérieure sera comparée.

---

## Méthode

### Corpus

27 articles de Wikipédia en français (IA, recherche d'information, bases de
données), **épinglés à une révision précise** dans
[`backend/eval/sources.json`](../backend/eval/sources.json) : ~1 M de caractères,
**1312 chunks** avec la configuration par défaut. Le texte est versionné, donc
l'évaluation tourne hors-ligne et les scores ne bougent pas parce qu'un
contributeur a modifié un article.

```bash
python -m eval.fetch_corpus     # reproduit le corpus à l'identique
```

### Golden set

31 questions ([`backend/eval/golden_set.json`](../backend/eval/golden_set.json)),
chacune associée à un **extrait verbatim** du corpus.

> **Un chunk est pertinent s'il contient l'extrait attendu.**

Le point important est ce que la référence ne contient *pas* : des identifiants
de chunks. Annoter « la question Q attend le chunk #42 » aurait rendu le golden
set caduc dès le premier changement de `chunk_size`, puisque les chunks sont
alors recréés avec d'autres frontières — et l'A/B testing du découpage serait
devenu impossible. La vérité terrain est donc **dérivée** à chaque exécution en
cherchant les extraits parmi les chunks réellement indexés.

Les questions sont étiquetées :

| Type | Ce qu'il teste | Nombre |
|---|---|---|
| `semantic` | Formulation différente du texte source — le lexical seul doit peiner | 19 |
| `lexical` | Terme exact, sigle ou nom propre — le bi-encodeur est typiquement plus faible | 12 |

Cette séparation est délibérée : elle permettra à la comparaison hybride de dire
quelque chose de précis plutôt que de déplacer un chiffre agrégé.

Deux tests protègent l'ensemble : chaque extrait doit exister verbatim dans son
document, et survivre au découpage (un extrait plus long que le chevauchement
pourrait tomber à cheval sur deux chunks et n'appartenir à aucun).

### Métriques

Implémentées à la main dans [`backend/eval/metrics.py`](../backend/eval/metrics.py) —
quelques lignes chacune, et ce sont elles qu'on raisonne :

| Métrique | Définition | Pourquoi |
|---|---|---|
| `hit@k` | au moins un chunk pertinent dans le top k | Le plus proche du besoin réel : le LLM n'a besoin que d'**un** bon passage |
| `doc_hit@k` | au moins un chunk du **bon document** dans le top k | Sépare « mauvais sujet » de « bon article, mauvais passage » |
| `recall@k` | part des chunks pertinents retrouvés | Exhaustivité |
| `precision@k` | part du top k qui est pertinente | Coût en contexte |
| `MRR` | 1 / rang du premier pertinent | Récompense de placer le bon passage **en tête** |
| `nDCG@k` | gain cumulé actualisé, normalisé | Sensible au rang *et* à tous les pertinents |

Toutes sont déterministes et ne coûtent aucun appel LLM : elles peuvent donc
servir de garde-fou anti-régression en CI, contrairement aux métriques jugées
par un LLM (fidélité, pertinence de la réponse), qui restent hors CI.

---

## Résultats de référence

Configuration : `chunk_size=1024`, `chunk_overlap=200`, k=5, embeddings
`paraphrase-multilingual-MiniLM-L12-v2` (384d), reranking
`mmarco-mMiniLMv2-L12-H384-v1`. Latences mesurées après préchauffage des modèles.

| Configuration | hit@5 | doc_hit@5 | recall@5 | precision@5 | MRR | nDCG@5 | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| Vectoriel seul | 0.419 | 0.710 | 0.342 | 0.090 | 0.309 | 0.286 | 22 ms | 25 ms |
| **Vectoriel + reranking** | **0.548** | **0.774** | **0.422** | **0.123** | **0.492** | **0.411** | 560 ms | 704 ms |

Par type de question :

| Configuration | Type | hit@5 | doc_hit@5 | MRR |
|---|---|---|---|---|
| Vectoriel seul | lexical | 0.500 | 0.833 | 0.403 |
| Vectoriel seul | semantic | 0.368 | 0.632 | 0.250 |
| Vectoriel + reranking | lexical | 0.750 | 0.917 | 0.708 |
| Vectoriel + reranking | semantic | 0.421 | 0.684 | 0.355 |

Reproduire :

```bash
cd backend && python -m eval.runner --compare
```

---

## Lecture des résultats

**Le reranking apporte un gain net et chiffré.** hit@5 +31 %, MRR +59 %,
nDCG@5 +44 % en relatif. Le gain sur le MRR est plus fort que sur le hit@5 :
le cross-encoder ne trouve pas seulement plus de bons passages, il les **remonte
en tête**, ce qui compte pour un LLM qui lit mieux le haut de son contexte.

**Il coûte cher.** 22 ms → 560 ms en médiane, soit ×25. C'est le compromis à
assumer : sur ce corpus il est rentable, mais la décision n'est pas gratuite et
elle est désormais explicite (`RERANK_ENABLED`).

**L'écart entre `hit@5` (0.55) et `doc_hit@5` (0.77) est le résultat le plus
utile.** Dans ~22 % des cas, le bon *article* est récupéré mais pas le passage
qui contient la réponse. Ce n'est donc pas un problème de compréhension du sujet
mais de **granularité du découpage** — ce qui désigne directement l'A/B testing
du chunking comme piste, plutôt qu'un changement de modèle d'embeddings.

**Les questions à terme exact restent le point faible du vectoriel.** Exemple
concret : « Quel article scientifique a présenté le transformeur ? », dont la
réponse est le titre anglais *Attention Is All You Need*. La recherche
vectorielle classe le bon chunk **au 37ᵉ rang** et remonte à la place des
passages génériques sur l'IA. Aucun réglage de seuil ne rattrape cela — c'est
structurellement ce qu'une recherche lexicale résout, et c'est la justification
chiffrée du chantier hybride.

---

## Limites assumées

- **31 questions** : suffisant pour comparer des configurations, trop peu pour
  des intervalles de confiance. Un écart de quelques points n'est pas
  significatif ; les écarts rapportés ici sont de l'ordre de 30 à 60 %.
- **`precision@k` est bornée par construction.** La plupart des questions n'ont
  qu'un à trois chunks pertinents, donc `precision@5` ne peut mécaniquement pas
  dépasser 0.2 à 0.6. Elle sert à comparer des configurations entre elles, pas à
  être lue dans l'absolu.
- **La pertinence est stricte et au niveau du chunk.** Un chunk du bon article
  qui reformule la réponse sans contenir l'extrait attendu compte comme un
  échec. C'est délibéré — pour répondre il faut le passage, pas le voisinage —
  mais cela tire les scores absolus vers le bas. C'est précisément ce que
  `doc_hit@k` permet de distinguer.
- **La génération n'est pas évaluée ici.** Ce document ne mesure que la
  récupération, dont tout le reste dépend. Les métriques de fidélité
  (*faithfulness*) jugées par LLM viendront séparément, hors CI.
