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

## Résultats

Configuration : `chunk_size=1024`, `chunk_overlap=200`, k=5, embeddings
`paraphrase-multilingual-MiniLM-L12-v2` (384d), reranking
`mmarco-mMiniLMv2-L12-H384-v1`, fusion RRF (k=60). Latences mesurées après
préchauffage des modèles.

| Configuration | hit@5 | doc_hit@5 | recall@5 | precision@5 | MRR | nDCG@5 | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| Vectoriel seul | 0.419 | 0.710 | 0.342 | 0.090 | 0.309 | 0.286 | 22 ms | 25 ms |
| Vectoriel + reranking | 0.581 | 0.774 | 0.454 | 0.129 | 0.524 | 0.443 | 577 ms | 701 ms |
| Hybride seul | 0.645 | 0.806 | 0.517 | 0.135 | 0.346 | 0.372 | 44 ms | 67 ms |
| **Hybride + reranking** | **0.774** | **0.903** | **0.664** | **0.181** | **0.664** | **0.614** | 1151 ms | 1366 ms |

Par type de question :

| Configuration | Type | hit@5 | doc_hit@5 | MRR |
|---|---|---|---|---|
| Vectoriel seul | lexical | 0.500 | 0.833 | 0.403 |
| Vectoriel seul | semantic | 0.368 | 0.632 | 0.250 |
| Hybride + reranking | **lexical** | **1.000** | **1.000** | **0.917** |
| Hybride + reranking | semantic | 0.632 | 0.842 | 0.504 |

Reproduire :

```bash
cd backend && python -m eval.runner --compare
```

---

## Lecture des résultats

**Les deux étages font des choses différentes, et c'est mesurable.**

- **L'hybride trouve** : à lui seul il fait mieux que le vectoriel + reranking
  sur `hit@5` (0.645 contre 0.581) **pour 13 fois moins de latence** (44 ms
  contre 577 ms). Mais son MRR est bien plus faible (0.346 contre 0.524) : il
  ramène les bons passages sans savoir les ordonner.
- **Le reranking ordonne** : il gagne peu en `hit@5` mais beaucoup en MRR, parce
  qu'il remonte le bon passage en tête — ce qui compte pour un LLM qui lit mieux
  le haut de son contexte.

Les deux sont donc complémentaires, et c'est leur combinaison qui paie :
`hit@5` +85 %, MRR +115 %, nDCG@5 +115 % par rapport au vectoriel seul.

**Le point aveugle des termes exacts est fermé.** Les 12 questions lexicales
passent toutes (`hit@5` = 1.000, `doc_hit@5` = 1.000), contre 6 sur 12 pour le
vectoriel seul. C'était l'hypothèse de départ du chantier hybride ; elle est
vérifiée.

**L'hybride aide aussi les questions sémantiques** (0.368 → 0.632), ce qui
n'allait pas de soi : même reformulée, une question française partage souvent
assez de vocabulaire avec sa réponse pour que le lexical contribue.

**Ce qui reste à traiter.** Les 7 échecs restants sont tous sémantiques, et
l'écart persistant entre `hit@5` (0.774) et `doc_hit@5` (0.903) dit que le bon
article est trouvé dans 13 % des cas sans le passage qui répond — un problème de
granularité de découpage, donc l'objet de l'A/B testing du chunking.

### Deux réglages découverts en mesurant

**Le budget du reranker doit être séparé du pool par moteur.** La fusion de deux
classements de 20 produit jusqu'à 40 candidats ; les tronquer à 20 avant le
cross-encoder jetait l'essentiel de l'apport lexical avant tout jugement. En
passant le plafond à 40 : `hit@5` 0.710 → 0.774, mais la latence double
(594 → 1151 ms), le cross-encoder notant deux fois plus de passages. Compromis
assumé et réglable (`RERANK_MAX_CANDIDATES`).

**Le OU lexical n'est pas un détail.** `websearch_to_tsquery` et
`plainto_tsquery` combinent les termes en **ET** : sur une question en langage
naturel, aucun chunk ne contient tous les mots de contenu, et la recherche
lexicale renvoie **zéro résultat**. Les lexèmes sont donc combinés en OU, ce qui
en refait un problème de classement plutôt qu'un filtre.

---

## Un incident de méthode, et sa correction

La première version du golden set demandait quel article a présenté le
transformeur, en attendant le titre *Attention Is All You Need*. La question
échouait dans **les quatre** configurations, y compris avec le lexical, ce qui
était suspect.

Diagnostic : dans le corpus, cette chaîne n'apparaît **que** dans une infobox et
dans des listes de références bibliographiques — jamais dans de la prose. Tous
les chunks marqués « pertinents » étaient donc des blocs de citations, qu'un
cross-encoder classe à juste titre comme de mauvaises réponses. La question
pénalisait le moteur pour s'être bien comporté.

C'est une limite intrinsèque d'une vérité terrain définie par sous-chaîne : elle
ne distingue pas un passage qui **répond** d'un passage qui **mentionne**. La
question a été remplacée par une autre, réellement ancrée dans de la prose. Le
garde-fou est de vérifier qu'un échec systématique vient bien du moteur, et non
de l'annotation.

---

## Limites assumées

- **31 questions** : suffisant pour comparer des configurations, trop peu pour
  des intervalles de confiance. Un écart de quelques points n'est pas
  significatif ; les écarts rapportés ici sont de l'ordre de 30 à 60 %.
- **`precision@k` est bornée par construction.** La plupart des questions n'ont
  qu'un à trois chunks pertinents, donc `precision@5` ne peut mécaniquement pas
  dépasser 0.2 à 0.6. Elle sert à comparer des configurations entre elles, pas à
  être lue dans l'absolu.
- **Une vérité terrain par sous-chaîne ne distingue pas répondre de
  mentionner** (voir l'incident ci-dessus). Les questions dont la réponse
  n'existe que dans une bibliographie ou une infobox sont à écarter à
  l'écriture du golden set.
- **La pertinence est stricte et au niveau du chunk.** Un chunk du bon article
  qui reformule la réponse sans contenir l'extrait attendu compte comme un
  échec. C'est délibéré — pour répondre il faut le passage, pas le voisinage —
  mais cela tire les scores absolus vers le bas. C'est précisément ce que
  `doc_hit@k` permet de distinguer.
- **La génération n'est pas évaluée ici.** Ce document ne mesure que la
  récupération, dont tout le reste dépend. Les métriques de fidélité
  (*faithfulness*) jugées par LLM viendront séparément, hors CI.
- **Le classement lexical n'est pas BM25.** `ts_rank_cd` n'intègre pas de
  fréquence inverse de document : un terme rare n'est pas privilégié sur un
  terme courant. Du vrai BM25 demanderait une extension (ParadeDB,
  `pg_search`) et alourdirait le déploiement.
- **La recherche lexicale est sensible aux accents.** `to_tsvector('french')`
  applique la racinisation française mais ne retire pas les accents ; une
  requête tapée sans accent ne matchera pas. Y remédier suppose de marquer
  `unaccent` comme IMMUTABLE, ce qu'il n'est pas — mentir là-dessus peut
  corrompre l'index si le dictionnaire change.
