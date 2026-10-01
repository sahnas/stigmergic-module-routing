# Préenregistrement, ablation du mécanisme

Écrit le 2026-10-01T04:57Z, avant tout lancement sur les graines de test.
Code : exp_mecanisme.py (sha256 02670360a17ee25b), exp_dynamique.py (sha256 c5f0595ecf9201ca), exp.py (sha256 9b0d44c40e14df1d).
Une graine de développement (400) pour vérifier que le code tourne, sans réglage. Aucun hyperparamètre modifié.
Graines de test : 80 à 99 (20, jamais utilisées). Variantes : aco_seuils, aco_seuils_hasard, aco_seuil_global.

Hypothèses (mesure : disponibilité) :
- M1 : recruter l'unité la moins occupée bat le recrutement au hasard (aco_seuils > aco_seuils_hasard).
- M2 : des seuils propres à chaque unité battent un seuil collectif unique (aco_seuils > aco_seuil_global).
Test : Wilcoxon apparié bilatéral, seuil 0,025 par test. Confirmée si p < 0,025 et différence > 0 ;
contredite si p < 0,025 et différence < 0 ; indéterminée sinon.
