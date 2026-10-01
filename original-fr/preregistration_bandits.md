# Préenregistrement, adversaires bandits non stationnaires

Écrit le 2026-10-01T04:01Z, avant le réglage et avant les graines de test.

Code : bandits.py (sha256 1e018844e58d6497), exp_dynamique.py (sha256 c5f0595ecf9201ca), exp.py (sha256 9b0d44c40e14df1d).
Même scénario de production que le test précédent.

Adversaires : SW-UCB, D-UCB, M-UCB (un bandit par symbole, bras = modules, récompense = q).
Réglage des adversaires, et d'eux seuls, sur les graines de développement 300, 301, 302 :
- SW-UCB : fenêtre {50, 200, 800} x c {0,05 ; 0,3}
- D-UCB : gamma {0,98 ; 0,995 ; 0,999} x c {0,05 ; 0,3}
- M-UCB : fenêtre {20, 60} x seuil {0,2 ; 0,4} x c {0,05 ; 0,3}
Critère : meilleure disponibilité moyenne sur ces 3 graines, une configuration retenue par algorithme.
Évaporation + seuils (aco_seuils) garde ses hyperparamètres d'origine, sans aucun réglage.

Graines de test : 60 à 79 (20 graines, jamais utilisées).
Hypothèses :
- B1 : aco_seuils > meilleur SW-UCB en disponibilité.
- B2 : aco_seuils > meilleur D-UCB en disponibilité.
- B3 : aco_seuils > meilleur M-UCB en disponibilité.
Test : Wilcoxon apparié bilatéral, seuil 0,05 / 3 = 0,0167, et différence moyenne positive.
La thèse « bat les bandits non stationnaires » n'est retenue que si B1, B2 et B3 sont confirmées.
Tout autre résultat est rapporté tel quel.

## Addendum, écrit le 2026-10-01T04:25Z

Une exécution antérieure, interrompue, a lancé ce protocole sur les graines 60 à 74 (fichier test_bandits.jsonl)
puis a commencé un adversaire supplémentaire non préenregistré (exp_rupture.py, reglage.jsonl), écarté.
Je n'ai pas examiné ces résultats avant d'écrire cet addendum.
Vérifications faites : bandits.py, exp_dynamique.py et exp.py ont les mêmes empreintes que ci-dessus ;
les configurations retenues (bandits_retenus.json) sont bien les meilleures sur les graines 300 à 302.
Pour éviter toute contamination, le test principal se fait sur des graines neuves : 80 à 99.
Même code, mêmes configurations, mêmes hypothèses B1 à B3, même seuil.
Les graines 60 à 74 seront rapportées à part, comme réplication.
