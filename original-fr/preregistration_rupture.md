# Préenregistrement, adversaire fort : bandit à détection de rupture

Écrit le 2026-10-01T04:43Z, avant tout lancement sur les graines de test.

Code : exp_rupture.py (sha256 abc0cf698ac27474), exp_dynamique.py (sha256 c5f0595ecf9201ca), exp.py (sha256 9b0d44c40e14df1d).
Scénario de production identique au test « sans gradient ».

Réglage de l'adversaire : 16 configurations sur les graines de développement 300, 301, 302 (grid_rupture_resume.txt).
Configuration retenue (meilleure disponibilité moyenne en développement, 0,619) : UCB, c = 0,1, h = 20, b = 0,2, fenêtre 300.
Le mécanisme testé (aco_seuils) n'est pas réglé : mêmes hyperparamètres depuis le premier test. Avantage délibéré à l'adversaire.

Graines de test : 60 à 79 (20 graines, jamais utilisées). Variantes : ema, aco_seuils, rupture.

Hypothèses :
- R1 : aco_seuils contre rupture en disponibilité, bilatéral. Thèse « sans gradient » maintenue si p < 0,025 et différence > 0 ;
  invalidée si p < 0,025 et différence < 0 ; indéterminée sinon.
- R2 : rupture > ema en disponibilité (l'adversaire est bien plus fort que le statu quo précédent), seuil 0,025.
Test : Wilcoxon apparié bilatéral. Descriptif : R² final, zero-shot final, disponibilité par événement, effondrements.
