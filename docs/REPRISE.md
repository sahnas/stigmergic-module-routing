# Reprise du projet JEPA / LeWorldModel

État scientifique au 13 octobre 2026. Ce fichier est le point d'entrée d'un
nouveau chat. Lire ensuite le [bilan](summary_two_pages.md), les résultats cités
ci-dessous et `git log -5 --oneline` pour connaître la révision utilisée.

## Ce qui est établi

- Expériences 18–19 : le prédicteur bilinéaire dépasse les références neurales
  sur Two-rooms avec la recette testée, mais leur est inférieur sur PushT.
  Une seule version entraînée par prédicteur et par tâche ; les quatre graines
  sont des graines d'évaluation. Le mécanisme reste inconnu.
- Expérience 20 : l'erreur dans l'état de tâche ne classe pas les prédicteurs
  comme la planification sur Two-rooms. Ce diagnostic ne prouve pas un mécanisme.
- [Expérience 21](../results/lewm_kaggle/exp21_selection/README.md) : 36
  évaluations terminées. Changer la sélection finale du CEM réduit l'écart de
  −1 / 0 / +1 points, contre au moins 5 requis sur chaque graine. Critère échoué.
- [Expérience 23](../results/lewm_kaggle/exp23_objective/README.md) : version 1
  arrêtée au contrôle du décodeur. Sur 89 618 observations de validation,
  médiane 3,339 ≤4 ; p95 8,977 >8. **Zéro évaluation de planification.**
  L'effet du nouvel objectif n'a pas été testé.
- [Audit du décodeur figé](../results/lewm_kaggle/exp23_decoder_audit/README.md) :
  terminé, exploratoire. Erreur >8 pour 52,93 % des observations dans la bande
  extérieure de 28 unités, contre 5,56 % dans l'intérieur. Le p95 intérieur
  reste à 8,207. Les erreurs touchent 889 des 969 épisodes de validation.
  Les 60 images examinées sont cohérentes avec les positions enregistrées.
  Ces associations ne déterminent pas la cause des erreurs de décodage.

## Décision suivante, non exécutée

L'expérience 22 reste un [brouillon](../preregistrations/22_planner_reachable_discrepancy_DRAFT.md).
Deux voies restent à décider : un protocole distinct de développement et de
validation indépendante du décodeur avant de retester l'objectif, ou le
diagnostic 22 sur les coûts des plans réellement considérés par le solveur.
L'audit ne démontre pas qu'un décodeur non linéaire résoudrait le problème.

La validation de l'expérience 23 a désormais été inspectée et devient une source
de développement pour toute nouvelle recette. Une confirmation exige des
épisodes indépendants non inspectés. Garder les exclusions des 388 épisodes de
planification, contrôler les embeddings prédits avant toute nouvelle comparaison
et ne pas modifier les seuils ou supprimer les observations gênantes pour
transformer l'échec de la version 1 en réussite. Aucun run ne part automatiquement.

## Vérifier et publier

```sh
git status --short
git log -5 --oneline
python tools/verify_exp23_freeze.py
python tools/verify_exp23_decoder_audit.py
python paper/check_numbers.py
```

Le [registre de publication](exp23_publication.md) conserve l'écriture GitHub
initialement refusée et la publication ultérieure à `f10ab04`, après les runs.
Ne pas présenter cette publication tardive comme un horodatage public avant
exécution. Les fichiers figés et les anciens statuts de soumission restent des
preuves historiques ; les READMEs de résultats donnent l'état scientifique actuel.

Le manuscrit et son PDF incluent maintenant les suivis 20, 21 et 23. Toute
nouvelle modification du LaTeX exige de régénérer le PDF correspondant.

Les clés d'accès ne font pas partie du contexte à conserver. Pour reprendre,
donner le dépôt et demander : « Lis `docs/REPRISE.md`, vérifie la révision,
puis reprends la décision suivante sans relancer une expérience. »
