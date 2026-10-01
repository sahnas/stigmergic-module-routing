# Préenregistrement, thèse « coordination sans gradient »

Écrit le 2026-09-30T23:27Z, avant tout lancement sur les graines de test.

Code : exp_dynamique.py (sha256 c5f0595ecf9201ca) et exp.py (sha256 9b0d44c40e14df1d).
Aucun hyperparamètre modifié par rapport au premier test. Une seule graine de développement (200), utilisée pour vérifier que le code tourne, sans réglage.
Graines de test : 40 à 59 (20 graines, jamais utilisées).

Plan factoriel 2 x 2 : règle d'oubli (ema = récence, aco = évaporation) x seuils par unité (non / oui).
Mesure principale : disponibilité = R² moyen sur les 17 tâches d'entraînement, évalué après chaque bloc
pendant toute la phase de production (4 événements : mort, arrivée + mort, dérive du monde, double mort).

Hypothèses :
- S1 : aco_seuils > ema en disponibilité (le mécanisme complet contre le statu quo).
- S2 : effet des seuils > 0, mesuré par graine comme ((ema_seuils - ema) + (aco_seuils - aco)) / 2.
- S3 : effet de la règle d'oubli, ((aco - ema) + (aco_seuils - ema_seuils)) / 2, bilatéral, sans prédiction de signe.

Test : Wilcoxon apparié (ou signé sur les différences) bilatéral, seuil 0,05 / 3 = 0,0167.
Critère : p < 0,0167 et différence moyenne dans le sens annoncé. Tout autre résultat est rapporté comme non confirmé.
Descriptif, hors test : R² final, zero-shot final, disponibilité par événement, nombre d'alarmes, effondrements.
