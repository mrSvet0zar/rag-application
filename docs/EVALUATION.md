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

Configuration retenue : `chunk_size=512`, `chunk_overlap=100` (voir le sweep
plus bas), k=5, embeddings `paraphrase-multilingual-MiniLM-L12-v2` (384d),
reranking `mmarco-mMiniLMv2-L12-H384-v1`, fusion RRF (k=60). Latences mesurées
après préchauffage des modèles.

| Configuration | hit@5 | doc_hit@5 | recall@5 | precision@5 | MRR | nDCG@5 | p50 | p95 |
|---|---|---|---|---|---|---|---|---|
| Vectoriel seul | 0.387 | 0.742 | 0.357 | 0.084 | 0.290 | 0.296 | 27 ms | 31 ms |
| Vectoriel + reranking | 0.613 | 0.903 | 0.568 | 0.135 | 0.478 | 0.477 | 293 ms | 359 ms |
| Hybride seul | 0.484 | 0.806 | 0.455 | 0.110 | 0.324 | 0.345 | 64 ms | 91 ms |
| **Hybride + reranking** | **0.806** | **0.935** | **0.735** | **0.174** | **0.652** | **0.635** | 574 ms | 719 ms |

Par type de question, dans la configuration retenue :

| Type | hit@5 | doc_hit@5 | MRR | (vectoriel seul, hit@5) |
|---|---|---|---|---|
| lexical (12) | 0.917 | **1.000** | 0.875 | 0.500 |
| semantic (19) | 0.737 | 0.895 | 0.511 | 0.316 |

Reproduire :

```bash
cd backend && python -m eval.runner --compare
```

---

## Lecture des résultats

**Les deux étages font des choses différentes.** L'hybride **trouve** : à
découpage large (1024) il battait à lui seul le vectoriel + reranking sur
`hit@5` pour 13 fois moins de latence. Le reranking **ordonne** : peu de gain en
`hit@5`, beaucoup en MRR, parce qu'il remonte le bon passage en tête — ce qui
compte pour un LLM qui lit mieux le haut de son contexte. C'est leur combinaison
qui paie : `hit@5` +108 %, MRR +125 %, nDCG@5 +115 % sur le vectoriel seul.

**Le point aveugle des termes exacts est fermé.** Les questions lexicales
atteignent `doc_hit@5` = 1.000 et `hit@5` = 0.917, contre 0.500 pour le
vectoriel seul. C'était l'hypothèse de départ du chantier hybride ; elle est
vérifiée.

**L'hybride aide aussi les questions sémantiques** (0.316 → 0.737), ce qui
n'allait pas de soi : même reformulée, une question française partage assez de
vocabulaire avec sa réponse pour que le lexical contribue.

### Réglages découverts en mesurant

**Le budget du reranker doit être séparé du pool par moteur.** La fusion de deux
classements de 20 produit jusqu'à 40 candidats ; les tronquer à 20 avant le
cross-encoder jetait l'essentiel de l'apport lexical avant tout jugement. Le
plafond porté à 40 : `hit@5` 0.710 → 0.774 (à 1024/200), au prix du double de
latence. Compromis assumé et réglable (`RERANK_MAX_CANDIDATES`).

**Le OU lexical n'est pas un détail.** `websearch_to_tsquery` et
`plainto_tsquery` combinent les termes en **ET** : sur une question en langage
naturel, aucun chunk ne contient tous les mots de contenu, et la recherche
lexicale renvoie **zéro résultat**. Les lexèmes sont donc combinés en OU, ce qui
en refait un problème de classement plutôt qu'un filtre.

---

## A/B testing du découpage

Le diagnostic de la baseline pointait le découpage : le bon *article* était
souvent trouvé sans le passage qui répond. Quatre configurations comparées, à
retrieval constant (hybride + reranking).

```bash
cd backend && python -m eval.runner --sweep
```

**À k constant (k=5)**

| chunk / overlap | hit@5 | doc_hit@5 | MRR | nDCG@5 | p50 |
|---|---|---|---|---|---|
| 256 / 50 | 0.742 | 0.871 | 0.605 | 0.567 | 371 ms |
| **512 / 100** | **0.806** | **0.935** | 0.652 | 0.635 | 624 ms |
| 1024 / 200 | 0.774 | 0.903 | 0.664 | 0.614 | 1177 ms |
| 2048 / 400 | 0.806 | 0.839 | **0.748** | **0.704** | 1990 ms |

**À budget de contexte constant** (~5120 caractères, k ajusté en conséquence)

| chunk / overlap | k | hit@k | doc_hit@k | MRR | nDCG@k | p50 |
|---|---|---|---|---|---|---|
| 256 / 50 | 20 | 0.742 | 0.871 | 0.605 | 0.562 | 347 ms |
| **512 / 100** | 10 | **0.839** | **0.935** | 0.657 | 0.640 | 622 ms |
| 1024 / 200 | 5 | 0.774 | 0.903 | 0.664 | 0.614 | 1183 ms |
| 2048 / 400 | 2 | 0.774 | 0.839 | 0.742 | 0.700 | 2055 ms |

> ⚠️ **Toutes les métriques ne sont pas comparables entre découpages.**
> `recall@k` et `precision@k` sont relatifs au nombre de chunks pertinents, et
> ce nombre change avec le découpage. Seuls `hit@k`, `doc_hit@k` et `MRR` se
> comparent directement. C'est aussi pourquoi la seconde vue existe : à k
> constant, cinq chunks de 256 caractères donnent au LLM quatre fois moins de
> texte que cinq chunks de 1024 — la comparaison serait faussée.

**512/100 est retenu.** Il est le meilleur sur les deux métriques comparables
dans les deux vues, et **divise la latence par deux** face à 1024/200 : le
cross-encoder note toujours 40 passages, mais deux fois plus courts.

**Le découpage interagit avec le reranking**, ce qu'aucune des deux mesures
seules n'aurait montré. En passant de 1024 à 512 :

| | 1024/200 | 512/100 |
|---|---|---|
| Hybride **seul** | 0.645 | 0.484 |
| Hybride **+ reranking** | 0.774 | **0.806** |

Des chunks courts dégradent la récupération brute — chacun porte moins de
contexte, donc les signaux dense et lexical sont plus fragmentés — mais donnent
au cross-encoder des passages plus précis à juger. Choisir un découpage sans
tenir compte de l'étage suivant aurait conduit à la mauvaise conclusion.

**2048/400 a le meilleur MRR et nDCG** mais le pire `doc_hit@5` (0.839) et une
latence de 2 s. De gros chunks contiennent plus, donc un passage trouvé est plus
souvent classé premier ; mais ils ratent plus souvent le bon article, et
saturent le contexte du LLM avec du texte non pertinent.

> 📌 **Conséquence opérationnelle** : changer le découpage ne re-découpe pas les
> documents déjà indexés. Une base existante conserve son ancien découpage
> jusqu'à ré-ingestion, et cohabite donc avec le nouveau.

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
