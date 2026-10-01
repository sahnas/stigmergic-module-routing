# Test minimal : coordination stigmergique, oubli des pistes, seuils par unité

> **Correction importante (1er octobre 2026, avant publication).** En préparant ce dépôt, j'ai trouvé dans l'environnement d'exécution les fichiers d'un protocole préenregistré que je n'avais pas rapporté : un test contre trois bandits non stationnaires standards (voir la section « Protocole retrouvé » en fin de document). Résultat : le mécanisme ne bat pas SW-UCB ni D-UCB (écarts non significatifs) ; il ne bat que les bandits à détection de rupture. La thèse « bat les bandits non stationnaires », telle que préenregistrée, n'est donc pas retenue. Par ailleurs, les graines 60 à 79 et 80 à 99, annoncées comme « jamais utilisées » dans deux préenregistrements, avaient déjà servi à ce protocole. Aucun choix n'a été informé par ces résultats, inconnus à ce moment-là, et le code est déterministe (vérifié : résultats identiques sur 20 graines), mais l'affirmation était fausse.

## Ce qui est testé, et ce qui ne l'est pas

Le mécanisme de coordination et de compensation, isolé. Huit petites unités (régresseurs supervisés, pas des JEPA) communiquent par l'espace d'observation, une interface fixe à la manière du flux texte Unix. Pas de latent partagé.

Monde : 5 primitives non linéaires en dimension 8. Une tâche est une primitive seule ou la composition de deux. Le système reçoit les symboles de la tâche, il ne découvre pas la décomposition (limite principale, voir plus bas).

Phase 1, apprentissage continu : 17 tâches présentées par blocs de 150 pas, deux passages dans un ordre aléatoire. Mesures : rétention sur les tâches d'entraînement, généralisation zero-shot sur 8 compositions jamais vues.

Phase 2, dommage : l'unité qui porte la primitive k* meurt définitivement (sortie nulle, ne réapprend pas), puis 2040 pas de récupération. Mesures : tâches impliquant k*, compositions tenues à l'écart impliquant k*, reste des tâches.

| Variante | Mécanisme |
|---|---|
| mono | MLP monolithique qui reçoit x et les symboles |
| oracle | orchestration fixe : symbole k vers unité k, pour toujours |
| soft | routeur appris par gradient (mélange softmax, façon mixture of experts) |
| cumul | traces symbole vers unité, moyenne cumulée, rien ne s'efface |
| ema | oubli par récence, sur l'arête utilisée seulement |
| aco | évaporation de toutes les arêtes du symbole sollicité + dépôt sur l'arête utilisée |
| global | évaporation de toutes les arêtes à chaque pas (fourmis au sens littéral) |
| aco_seuils | aco + seuil d'alarme propre à chaque unité + recrutement de l'unité la moins engagée |

Hypothèses fixées avant les graines de test 0 à 9 (mise au point sur 100 à 102). Hyperparamètres des variantes à traces choisis a priori, non réglés. Taux d'apprentissage vérifié pour les baselines. Contrôle : le MLP monolithique entraîné en iid atteint R² 0,93 sur les tâches d'entraînement mais reste négatif en zero-shot.

## Relancer

```
pip install torch numpy scipy matplotlib
python exp.py --seeds 0,1,2,3,4,5,6,7,8,9 --out results.jsonl
python analyse.py results.jsonl
```

Environ 70 s par graine sur un seul CPU.

## Résultats, 10 graines de test (R², 1 = parfait, 0 = prédit la moyenne)

| Variante | Rétention phase 1, moy. / méd. | Effondrements (< 0,5) | Zero-shot, moy. / méd. | k* après récupération | Reroutage, pas (méd.) | Temps / graine |
|---|---|---|---|---|---|---|
| mono | -0,47 / -0,46 | 10/10 | -0,50 / -0,50 | -0,05 | n/a | 7,0 s |
| oracle | 0,94 / 0,97 | 0/10 | 0,91 / 0,96 | 0,00 | jamais | 5,3 s |
| soft | 0,81 / 0,86 | 0/10 | 0,73 / 0,82 | 0,98 | 324 | 24,3 s |
| cumul | 0,75 / 0,96 | 2/10 | 0,70 / 0,94 | 0,00 | jamais | 6,5 s |
| ema | 0,76 / 0,95 | 2/10 | 0,68 / 0,94 | 0,75 | 524 | 6,7 s |
| aco | 0,83 / 0,94 | 2/10 | 0,72 / 0,92 | 0,92 | 555 | 6,8 s |
| global | 0,07 / 0,08 | 10/10 | -0,09 / -0,04 | 0,13 | 37 | 6,5 s |
| aco_seuils | 0,91 / 0,92 | 0/10 | 0,88 / 0,88 | 0,97 | 4,5 | 7,0 s |

Tests préenregistrés, Wilcoxon apparié bilatéral, seuil de Bonferroni 0,005 :

- H1 compensation : aco récupère, cumul et oracle jamais (10/10 graines, p = 0,002). Validée.
- H2 seuils : aco_seuils récupère mieux et reroute environ 100 fois plus vite que aco (10/10, p = 0,002). Validée.
- H3 oubli temporel global : détruit la rétention (0,07 contre 0,83, p = 0,002). Validée.
- H4 généralisation : aco > mono (p = 0,002) ; aco contre soft non significatif (p = 0,43). Partielle.
- Contrôle aco contre ema : aucune différence détectée (p = 0,56 et 1,0).

Exploratoire, non préenregistré, à confirmer sur de nouvelles graines : aco_seuils bat soft en zero-shot (0,88 contre 0,73) et en rétention (0,91 contre 0,81) sur 9/10 graines, p = 0,01, pour environ 3,5 fois moins de calcul. Les seuils déclenchent 24 à 42 fausses alarmes par run en fonctionnement normal.

## Limites

- Monde jouet : 5 primitives, dimension 8, une seule famille de mondes.
- Les symboles de la tâche sont donnés. Le vrai problème, découvrir la décomposition, n'est pas testé.
- Les unités ne sont pas des JEPA.
- Dommage favorable : une mort nette, avec trois unités de réserve disponibles.
- L'oubli du contenu des unités n'est pas testé.

## Confirmation préenregistrée (graines 20 à 29, code inchangé)

Voir preregistration_confirmation.md, écrit avant le lancement.

- C1, aco_seuils > soft en zero-shot : différence moyenne +0,008 (IC 95 % bootstrap -0,09 à +0,09), 7/10 graines, p = 0,63. Non confirmée.
- C2, aco_seuils > soft en rétention : différence moyenne +0,014 (IC 95 % -0,04 à +0,07), 7/10, p = 0,56. Non confirmée.

L'avantage exploratoire sur les graines 0 à 9 venait surtout de graines défavorables au routeur soft (rétention moyenne 0,81 sur 0 à 9, 0,88 sur 20 à 29). Sur ce monde jouet, le mécanisme fait jeu égal avec un routeur appris par gradient. Les résultats préenregistrés H1 à H3 ne sont pas concernés.

## Thèse « coordination sans gradient » (graines 40 à 59, préenregistrée)

Voir preregistration_sans_gradient.md et exp_dynamique.py. Le routage ne reçoit aucun gradient. Adversaire : routage par simple récence (ema). Scénario de production : mort d'un module, arrivée de deux modules et mort d'un autre, dérive du monde, double mort simultanée. 7 modules au départ sur 10 emplacements. Le système ne sait pas quels modules sont morts.

| Variante | Disponibilité (moy.) | Effondrements à l'apprentissage initial | R² final | Zero-shot final |
|---|---|---|---|---|
| récence (ema) | 0,41 | 10/20 | 0,22 | 0,07 |
| récence + seuils | 0,51 | 7/20 | 0,61 | 0,45 |
| évaporation (aco) | 0,57 | 4/20 | 0,37 | 0,20 |
| évaporation + seuils | 0,76 | 0/20 | 0,60 | 0,39 |

- S1, évaporation + seuils > récence : +0,35 (IC 95 % +0,29 à +0,40), 20/20 graines, p < 0,0001. Confirmée.
- S2, effet des seuils : +0,14 (IC 95 % +0,09 à +0,19), 18/20, p = 0,0001. Confirmée.
- S3, effet de la règle d'oubli (sans prédiction de signe) : évaporation > récence, +0,21 (IC 95 % +0,17 à +0,23), 20/20. Confirmée.

Descriptif : l'effet ne vient pas seulement de l'apprentissage initial. Sur les 7 graines où les deux variantes démarrent saines, évaporation + seuils garde +0,34 de disponibilité (7/7) et perd 0,30 de R² sur l'ensemble des événements, contre 0,77 pour la récence. Après la double mort, aucune variante ne récupère complètement ; les variantes sans seuils ne récupèrent pas du tout.

Limites propres à ce test : scénario conçu pour éprouver la résilience, donc non neutre ; adversaire raisonnable mais pas le plus fort (bandits à fenêtre glissante, UCB escompté, détection de rupture non testés) ; mêmes limites que plus haut (monde jouet, symboles donnés, pas de JEPA).

## Adversaire fort : bandit à détection de rupture (graines 60 à 79, préenregistré)

Voir preregistration_rupture.md, exp_rupture.py, grid_rupture_resume.txt. Bandit UCB par symbole, fenêtre glissante de 300, remise à zéro d'un bras quand sa qualité chute (famille M-UCB / CUSUM-UCB). L'adversaire a été réglé sur 16 configurations (graines 300 à 302) ; le mécanisme testé n'a jamais été réglé.

| Variante | Disponibilité (moy.) | Effondrements initiaux | R² initial | R² final | Zero-shot final |
|---|---|---|---|---|---|
| récence (ema) | 0,45 | 9/20 | 0,58 | 0,28 | 0,11 |
| bandit à détection de rupture | 0,61 | 0/20 | 0,95 | 0,60 | 0,39 |
| évaporation + seuils | 0,77 | 0/20 | 0,91 | 0,72 | 0,57 |

- R1, évaporation + seuils contre bandit réglé : +0,16 (IC 95 % +0,11 à +0,22), 18/20 graines, p < 0,0001. Thèse « sans gradient » maintenue.
- R2, le bandit bat la récence : +0,16 (IC 95 % +0,09 à +0,22), 18/20, p = 0,0009. Confirmée : l'adversaire est bien plus fort que le statu quo précédent.

Descriptif : le bandit apprend mieux au départ (0,95 contre 0,91) mais perd plus au fil des événements (0,37 contre 0,21 sur les 18 graines saines pour les deux). L'avantage porte sur la résilience, pas sur l'apprentissage.

Limites : un seul type d'adversaire (pas de GLR, ni de CUSUM au sens strict), réglage modeste (16 configurations, 3 graines), scénario conçu pour éprouver la résilience, monde jouet, symboles donnés, pas de JEPA.

## Ablation du mécanisme (graines 80 à 99, préenregistrée)

Voir preregistration_mecanisme.md et exp_mecanisme.py.

| Variante | Disponibilité (moy.) | R² final | Effondrements initiaux |
|---|---|---|---|
| évaporation + seuils individuels + recrutement du moins occupé | 0,74 | 0,65 | 0/20 |
| même chose, recrutement au hasard | 0,52 | 0,48 | 2/20 |
| même chose, un seul seuil collectif | 0,74 | 0,65 | 0/20 |

- M1, recruter le moins occupé plutôt qu'au hasard : +0,22 (IC 95 % +0,19 à +0,27), 20/20, p < 0,0001. Confirmée.
- M2, seuils individuels plutôt qu'un seuil collectif : +0,002 (IC 95 % -0,03 à +0,03), 12/20, p = 0,78. Indéterminée ; si un effet existe, il est inférieur à 0,03.

Ce qui porte le résultat : l'oubli déclenché par la sollicitation, une alarme sur chute anormale, et le recrutement fondé sur une information collective (la charge de chaque module). L'individualité des seuils n'apporte rien de mesurable dans ce scénario, où tous les modules sont identiques.

## Étape 2 : sans symboles (graines 600 à 609, préenregistrée)

Voir preregistration_sans_symboles.md et exp_sans_symboles.py. Chaque tâche n'a plus qu'un identifiant opaque ; le système doit découvrir que des tâches partagent des primitives. Phase A : 12 compositions connues, apprentissage entrelacé. Phase B : 8 compositions nouvelles, 200 pas chacune à partir du même état.

| Variante | R² tâches connues (phase A) | Vitesse sur nouvelle composition | R² final nouvelle | Interférence sur anciennes | Alignement modules / primitives |
|---|---|---|---|---|---|
| mono | 0,95 | 0,96 | 0,98 | 0,60 | n/a |
| soft (gradient) | 0,91 | 0,89 | 0,97 | 2,23 | -8,8 |
| rl (renforcement) | 0,13 | 0,29 | 0,50 | 0,21 | -0,24 |
| évaporation + seuils | 0,21 | 0,29 | 0,55 | 0,21 | -0,20 |

- N1 et N3 contredites : le mécanisme apprend les nouvelles compositions bien plus lentement que soft et mono (0/10 graines, p = 0,002).
- N2 indéterminée : jeu égal avec le routeur par renforcement.
- I1 et I2 confirmées mais non interprétables comme une qualité : l'interférence est faible parce que presque rien n'a été appris en phase A.
- Aucune variante n'aligne ses modules sur les primitives : personne ne découvre la décomposition. mono et soft apprennent chaque tâche comme une fonction propre, et écrasent les anciennes.

Conclusion : sur ce test, le mécanisme ne découvre pas la décomposition et n'apprend même pas les tâches connues sans symboles. Ses résultats positifs antérieurs reposaient sur une décomposition donnée. C'est un mécanisme de coordination et de résilience, pas de découverte de structure.


## Protocole retrouvé : bandits non stationnaires standards (préenregistré, graines 80 à 99, réplication 60 à 74)

Fichiers : preregistration_bandits.md (avec son addendum), bandits.py, tuning.jsonl, bandits_retenus.json, bandits_test2.jsonl, test_bandits.jsonl, analyse_bandits.py, resultats_bandits.txt. Ces fichiers proviennent d'une exécution antérieure absente de l'historique de la conversation qui a produit le reste du dépôt ; ils ont été retrouvés dans l'environnement d'exécution. Les empreintes du code correspondent au préenregistrement. Le test principal s'était arrêté à 15 graines sur 20 ; les graines 95 à 99 ont été complétées avec le même code avant toute analyse. Adversaires réglés sur les graines 300 à 302 (20 configurations) ; mécanisme non réglé.

| Variante | Disponibilité (moy.) | R² initial | R² final |
|---|---|---|---|
| évaporation + seuils | 0,74 | 0,94 | 0,65 |
| SW-UCB (fenêtre 50) | 0,72 | 0,69 | 0,86 |
| D-UCB (gamma 0,98) | 0,72 | 0,66 | 0,88 |
| M-UCB (détection de rupture) | 0,59 | 0,90 | 0,72 |

- B1, contre SW-UCB : +0,02 (IC 95 % -0,02 à +0,06), 13/20, p = 0,39. Non confirmée.
- B2, contre D-UCB : +0,03 (IC 95 % -0,01 à +0,06), 14/20, p = 0,20. Non confirmée.
- B3, contre M-UCB : +0,15 (IC 95 % +0,11 à +0,19), 20/20, p < 0,0001. Confirmée.
- Réplication (graines 60 à 74, 15 graines) : même tableau ; B1 et B2 positives mais non significatives au seuil préenregistré (p = 0,03), B3 confirmée.

Lecture : à disponibilité égale, les profils diffèrent. Le mécanisme apprend mieux au départ et encaisse mieux les premiers événements ; SW-UCB et D-UCB apprennent moins bien au départ mais récupèrent mieux à la fin, notamment après la double mort. La conclusion défendable est « au niveau des meilleurs bandits non stationnaires standards, meilleur que ceux à détection de rupture », pas « meilleur qu'eux ».

Réutilisation de graines, à déclarer : le test contre le bandit à détection de rupture (graines 60 à 79) et l'ablation (graines 80 à 99) ont été préenregistrés comme utilisant des graines neuves, alors que ces graines avaient servi à ce protocole. Leurs résultats restent valides au sens où aucune décision n'en dépendait, mais l'affirmation « jamais utilisées » était inexacte.
