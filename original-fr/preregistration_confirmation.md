# Préenregistrement, test de confirmation

Écrit le 2026-09-30T23:15Z, avant tout lancement sur les graines ci-dessous.

Code inchangé : exp.py, sha256 9b0d44c40e14df1d. Mêmes hyperparamètres, aucun réglage.
Graines nouvelles, jamais utilisées : 20 à 29.
Variantes : soft, aco_seuils (aco lancé pour référence, hors test).

Hypothèses (issues de l'analyse exploratoire sur les graines 0 à 9) :
- C1 : aco_seuils > soft en généralisation zero-shot (zero_r2).
- C2 : aco_seuils > soft en rétention fin de phase 1 (train_r2).

Test : Wilcoxon apparié bilatéral, seuil 0,025 par test (Bonferroni sur 2).
Critère de confirmation : p < 0,025 ET différence moyenne positive.
Résultat nul ou inversé = non confirmé, rapporté tel quel.
