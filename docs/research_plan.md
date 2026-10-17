# Plan de recherche

## Questions de recherche
- RQ1: Quel modèle classique obtient les meilleurs résultats sur chaque langue ?
- RQ2: Les performances se dégradent-elles lorsqu'un modèle est testé sur une langue différente de celle de l'entraînement ?
- RQ3: Un modèle multilingue pré-entraîné améliore-t-il la robustesse par rapport à TF-IDF ?
- RQ4: Quels types d'articles et quelles erreurs expliquent les échecs du modèle ?

## Expériences
1. Audit des données, vérification des licences et des labels.
2. Baselines séparées et multilingues : Naive Bayes, Linear SVM, Logistic Regression.
3. Évaluation macro-F1 et métriques par langue.
4. Évaluation cross-lingue (train sur une langue, test sur une autre) lorsque les labels sont comparables.
5. Comparaison avec un transformer multilingue si ressources suffisantes.
6. Analyse d'erreurs, biais de source, doublons et limites.

## Point méthodologique
Les performances annoncées dans l'article de référence ne doivent pas être présentées comme des résultats reproduits tant que le dataset, la version des données, les splits et la définition exacte de l'approche LR + MLE ne sont pas confirmés.
