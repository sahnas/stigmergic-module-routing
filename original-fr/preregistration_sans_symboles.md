# Préenregistrement, étape 2 : sans symboles

Écrit le 2026-10-01T05:18Z, avant tout lancement sur les graines de test.
Code : exp_sans_symboles.py (sha256 5c558d088ce911c3), exp.py (sha256 9b0d44c40e14df1d).

Développement : graines 500 et 501 pour vérifier le code et régler l'adversaire par renforcement
(lr_pg dans {0,1 ; 0,5 ; 2,0}, critère : vitesse d'apprentissage). Retenu : lr_pg = 0,1.
Le mécanisme testé (aco_seuils) n'est pas réglé.

Ce que les graines de développement ont déjà montré (à déclarer avant le test) :
aco_seuils apprend mal les 12 tâches connues sans symboles (R² 0,26 et 0,42) et apprend lentement
les nouvelles compositions ; mono et soft apprennent vite les nouvelles mais détruisent les anciennes.
Aucune variante n'aligne ses modules sur les primitives. Je m'attends donc à ce que N1 et N3 échouent.

Graines de test : 600 à 609 (10 graines, jamais utilisées). Variantes : mono, soft, rl, aco_seuils.

Famille principale (mesure définie dans le code avant le développement : vitesse d'apprentissage
d'une composition nouvelle, R² moyen sur 10 évaluations pendant 200 pas), seuil 0,05 / 3 :
- N1 : aco_seuils > soft
- N2 : aco_seuils > rl
- N3 : aco_seuils > mono
Famille secondaire (interférence sur les 12 anciennes tâches, plus bas = mieux), seuil 0,025 :
- I1 : aco_seuils < soft
- I2 : aco_seuils < mono
Wilcoxon apparié bilatéral. Confirmée si p sous le seuil et différence dans le sens annoncé ;
contredite si p sous le seuil dans l'autre sens ; indéterminée sinon.

Descriptif seulement, mesure ajoutée APRÈS avoir vu les graines de développement : capacité globale
= R² moyen tronqué à 0 sur les 13 tâches (12 anciennes + la nouvelle) après les 200 pas.
