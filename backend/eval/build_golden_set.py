"""Author `golden_set.json`.

Kept as a script rather than a hand-edited JSON file so the questions live next
to their rationale, and so the snippets are generated rather than retyped (the
corpus mixes both apostrophe characters). Run it from `backend/`:

    python -m eval.build_golden_set

Every entry pairs a question with *verbatim* snippets from the corpus. Relevance
is defined as "this chunk contains one of the snippets", never as "this chunk
id" — chunk ids change the moment chunking changes, which would make the A/B
chunking experiment impossible to run against a fixed golden set.

Snippets are kept short (well under the chunk overlap) so that a snippet is
always wholly contained in at least one chunk rather than straddling a boundary.

`kind` records what each question is meant to exercise:
  * "semantic" — worded differently from the source, so lexical matching alone
    should struggle;
  * "lexical"  — hinges on an exact term, acronym or proper noun, where a
    bi-encoder is typically weaker.
That split is deliberate: it is what will make the hybrid-search comparison
say something instead of moving one aggregate number.
"""

from __future__ import annotations

import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET = EVAL_DIR / "golden_set.json"

QUESTIONS: list[dict] = [
    # ---------- similarité cosinus ----------
    {
        "id": "q01",
        "question": "Comment obtient-on le cosinus de l'angle entre deux vecteurs ?",
        "document": "similarite_cosinus",
        "kind": "semantic",
        "snippets": ["produit scalaire divisé par le produit de leur norme"],
    },
    {
        "id": "q02",
        "question": "Que veut dire un score de similarité de -1 ?",
        "document": "similarite_cosinus",
        "kind": "semantic",
        "snippets": ["La valeur de -1 indique des vecteurs opposés"],
    },
    {
        "id": "q03",
        "question": "Entre quelles bornes varie la similarité cosinus ?",
        "document": "similarite_cosinus",
        "kind": "semantic",
        "snippets": ["comprise dans l'intervalle [-1,1]"],
    },
    {
        "id": "q04",
        "question": "Qu'est-ce que l'indice de Tanimoto ?",
        "document": "similarite_cosinus",
        "kind": "lexical",
        "snippets": [
            "indice de Tanimoto reprend cette idée dans le cas des attributs binaires"
        ],
    },
    # ---------- BM25 ----------
    {
        "id": "q05",
        "question": "Qui a proposé le modèle probabiliste de pertinence en 1976 ?",
        "document": "okapi_bm25",
        "kind": "lexical",
        "snippets": ["proposé en 1976 par Robertson et Jones"],
    },
    {
        "id": "q06",
        "question": "Pourquoi cette méthode de pondération s'appelle-t-elle Okapi ?",
        "document": "okapi_bm25",
        "kind": "lexical",
        "snippets": ["nom du système de recherche de l'université de Londres"],
    },
    {
        "id": "q07",
        "question": "Est-ce que l'ordre des mots compte dans ce classement de documents ?",
        "document": "okapi_bm25",
        "kind": "semantic",
        "snippets": ["modèle de sac de mots qui ordonne les documents"],
    },
    # ---------- TF-IDF ----------
    {
        "id": "q08",
        "question": "Que signifie le sigle TF-IDF ?",
        "document": "tf_idf",
        "kind": "lexical",
        "snippets": ["term frequency-inverse document frequency"],
    },
    {
        "id": "q09",
        "question": "Sur quelle loi statistique repose la pondération TF-IDF ?",
        "document": "tf_idf",
        "kind": "lexical",
        "snippets": ["donnée par la loi de Zipf"],
    },
    {
        "id": "q10",
        "question": "Pourquoi un mot très répandu apporte-t-il peu d'information ?",
        "document": "tf_idf",
        "kind": "semantic",
        "snippets": ["il est en fait peu discriminant"],
    },
    # ---------- précision et rappel ----------
    {
        "id": "q11",
        "question": "Quelle différence entre précision et rappel ?",
        "document": "precision_et_rappel",
        "kind": "semantic",
        "snippets": ["proportion d'items pertinents parmi les items sélectionnés"],
    },
    {
        "id": "q12",
        "question": "Comment nomme-t-on la précision en statistique ?",
        "document": "precision_et_rappel",
        "kind": "lexical",
        "snippets": ["précision est appelée valeur prédictive positive"],
    },
    {
        "id": "q13",
        "question": "Sur 30 pages retournées dont 20 pertinentes, que valent précision et rappel ?",
        "document": "precision_et_rappel",
        "kind": "semantic",
        "snippets": ["sa précision est de 20/(20+10)"],
    },
    {
        "id": "q14",
        "question": "Comment appelle-t-on les documents non pertinents renvoyés par un système ?",
        "document": "precision_et_rappel",
        "kind": "semantic",
        "snippets": ["non pertinents constituent du bruit"],
    },
    # ---------- plongement lexical ----------
    {
        "id": "q15",
        "question": "Comment représente-t-on des mots sous forme de vecteurs ?",
        "document": "plongement_lexical",
        "kind": "semantic",
        "snippets": ["représentation de mots sous forme de vecteurs"],
    },
    {
        "id": "q16",
        "question": "Sur quelle hypothèse linguistique reposent les plongements de mots ?",
        "document": "plongement_lexical",
        "kind": "semantic",
        "snippets": ["contextes similaires ont des significations apparentées"],
    },
    {
        "id": "q17",
        "question": "Quel est le terme français pour word embedding ?",
        "document": "plongement_lexical",
        "kind": "lexical",
        "snippets": ["plongement lexical"],
    },
    # ---------- validation croisée ----------
    {
        "id": "q18",
        "question": "Que se passe-t-il quand un modèle colle trop à ses données d'entraînement ?",
        "document": "validation_croisee",
        "kind": "semantic",
        "snippets": ["on parle de surapprentissage"],
    },
    {
        "id": "q19",
        "question": "Comment dit-on validation non croisée en anglais ?",
        "document": "validation_croisee",
        "kind": "lexical",
        "snippets": ["test set validation"],
    },
    # ---------- transformeur ----------
    {
        "id": "q20",
        "question": "En quelle année l'architecture transformeur a-t-elle été introduite ?",
        "document": "transformeur",
        "kind": "semantic",
        "snippets": ["apprentissage profond introduite en 2017"],
    },
    # An earlier q21 asked which paper introduced the transformer, expecting the
    # title "Attention Is All You Need". The corpus only carries that string in
    # an infobox and in bibliography entries — never in prose. Every "relevant"
    # chunk was therefore a citation list, which a reranker rightly scores as a
    # poor answer, so the question penalised the retriever for behaving well.
    # Replaced by a question the corpus answers in prose, on a document the
    # golden set did not otherwise cover.
    {
        "id": "q21",
        "question": "Que veut dire le sigle ACP en statistique ?",
        "document": "analyse_en_composantes_principales",
        "kind": "lexical",
        "snippets": ["ACP ou PCA en anglais pour principal component analysis"],
    },
    {
        "id": "q22",
        "question": "Pourquoi cette architecture s'entraîne-t-elle plus vite qu'un réseau récurrent ?",
        "document": "transformeur",
        "kind": "semantic",
        "snippets": ["ne nécessitent pas un traitement séquentiel des données"],
    },
    # ---------- attention ----------
    {
        "id": "q23",
        "question": "Quelle forme d'attention est la plus couramment employée ?",
        "document": "attention_apprentissage_automatique",
        "kind": "semantic",
        "snippets": ["produit scalaire pondéré"],
    },
    # ---------- grands modèles de langage ----------
    {
        "id": "q24",
        "question": "Que veut dire l'acronyme LLM ?",
        "document": "grand_modele_de_langage",
        "kind": "lexical",
        "snippets": ["large language model"],
    },
    # ---------- PostgreSQL ----------
    {
        "id": "q25",
        "question": "PostgreSQL garantit-il l'intégrité des données lors des transactions ?",
        "document": "postgresql",
        "kind": "semantic",
        "snippets": ["prend en charge les transactions ACID"],
    },
    # ---------- Docker ----------
    {
        "id": "q26",
        "question": "En quelle année Docker est-il apparu ?",
        "document": "docker_logiciel",
        "kind": "semantic",
        "snippets": ["mouture de Docker a été publiée en 2013"],
    },
    {
        "id": "q27",
        "question": "À quoi sert Docker ?",
        "document": "docker_logiciel",
        "kind": "semantic",
        "snippets": ["faire tourner certaines applications dans des conteneurs"],
    },
    # ---------- SQL ----------
    {
        "id": "q28",
        "question": "Que signifie le sigle SQL ?",
        "document": "structured_query_language",
        "kind": "lexical",
        "snippets": ["Structured Query Language"],
    },
    {
        "id": "q29",
        "question": "À quoi sert le langage utilisé pour interroger les bases relationnelles ?",
        "document": "structured_query_language",
        "kind": "semantic",
        "snippets": ["exploiter des bases de données relationnelles"],
    },
    # ---------- k plus proches voisins ----------
    {
        "id": "q30",
        "question": "Quel est le sigle anglais de la méthode des k plus proches voisins ?",
        "document": "methode_des_k_plus_proches_voisins",
        "kind": "lexical",
        "snippets": ["k-nearest neighbor"],
    },
    {
        "id": "q31",
        "question": "Comment cette méthode choisit-elle la classe d'un nouvel exemple ?",
        "document": "methode_des_k_plus_proches_voisins",
        "kind": "semantic",
        "snippets": ["la classe la plus représentée parmi les k sorties"],
    },
]


def main() -> int:
    payload = {
        "_comment": (
            "Golden set d'évaluation. La pertinence est définie par des extraits "
            "verbatim du corpus, jamais par des identifiants de chunks : les "
            "identifiants changent dès que la stratégie de découpage change. "
            "Généré par `python -m eval.build_golden_set`."
        ),
        "corpus": "eval/corpus",
        "questions": QUESTIONS,
    }
    GOLDEN_SET.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    kinds = {q["kind"] for q in QUESTIONS}
    counts = {kind: sum(1 for q in QUESTIONS if q["kind"] == kind) for kind in kinds}
    print(f"{len(QUESTIONS)} questions écrites dans {GOLDEN_SET.name} : {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
