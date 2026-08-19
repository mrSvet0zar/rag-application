# Corpus d'évaluation — attribution et licence

Les fichiers `.txt` de ce répertoire sont des extraits d'articles de
**Wikipédia en français** (https://fr.wikipedia.org), utilisés ici comme corpus
de référence pour mesurer la qualité de la recherche.

## Licence

Contenu disponible sous licence **Creative Commons Attribution — Partage dans
les Mêmes Conditions 4.0** (CC BY-SA 4.0) :
https://creativecommons.org/licenses/by-sa/4.0/deed.fr

Cette licence s'applique **uniquement à ce répertoire**. Le reste du dépôt est
sous licence MIT (voir [LICENSE](../../../LICENSE)).

## Provenance

Chaque fichier commence par le titre de l'article et le **numéro de révision**
exact dont il est extrait. Les révisions sont figées dans
[`../sources.json`](../sources.json), ce qui rend le corpus reproductible :

```bash
python -m eval.fetch_corpus          # reproduit ce corpus à l'identique
python -m eval.fetch_corpus --repin  # bascule sur les révisions courantes
```

L'historique complet des auteurs de chaque article est consultable via
`https://fr.wikipedia.org/w/index.php?oldid=<révision>&action=history`.

## Transformation appliquée

Le texte a été extrait du rendu HTML de l'article, puis nettoyé : suppression
des scripts, styles, menus, formules MathML, marqueurs de référence (`[1]`) et
liens « modifier ». Aucune modification du contenu rédactionnel.
