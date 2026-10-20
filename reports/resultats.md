# Résultats

45 211 clients contactés ; apprentissage sur les 31 647 premiers (taux de souscription 5,8 %), test sur les 13 564 suivants (25,4 %). Variables exclues : `duration` (fuite d'information) et `day`.

## Performance : validation hors temps contre découpage aléatoire

| validation   | modèle                             |   AUC |   Gini |    KS |   Brier | capture 10 %   | capture 20 %   |
|:-------------|:-----------------------------------|------:|-------:|------:|--------:|:---------------|:---------------|
| hors temps   | Grille de score (WoE + logistique) | 0,647 |  0,294 | 0,260 |   0,205 | 16 %           | 29 %           |
| hors temps   | XGBoost                            | 0,658 |  0,316 | 0,235 |   0,225 | 17 %           | 31 %           |
| aléatoire    | Grille de score (WoE + logistique) | 0,768 |  0,536 | 0,410 |   0,086 | 39 %           | 55 %           |
| aléatoire    | XGBoost                            | 0,744 |  0,488 | 0,348 |   0,100 | 36 %           | 49 %           |

Capture 20 % : part des souscripteurs touchés en n'appelant que les 20 % de clients les mieux scorés (20 % au hasard).

## Dérive entre apprentissage et test (PSI)

| variable                     |   PSI |
|:-----------------------------|------:|
| Mois du contact              | 1,822 |
| Canal de contact             | 1,368 |
| Résultat campagne précédente | 1,089 |
| Contacts antérieurs          | 0,977 |
| Jours depuis le contact      | 0,977 |
| Contacts (campagne)          | 0,155 |
| Profession                   | 0,116 |
| Âge                          | 0,107 |
| Prêt personnel               | 0,049 |
| Solde moyen                  | 0,047 |
| Situation familiale          | 0,042 |
| Défaut de crédit             | 0,020 |
| Prêt immobilier              | 0,006 |
| Études                       | 0,004 |

## Modèle figé contre réentraînement (fenêtre croissante), par bloc de la période de test

|   bloc | modèle                             | mode       |   AUC | taux observé   | taux prédit   |
|-------:|:-----------------------------------|:-----------|------:|:---------------|:--------------|
|      1 | Grille de score (WoE + logistique) | figé       | 0,631 | 15,5 %         | 25,6 %        |
|      1 | Grille de score (WoE + logistique) | réentraîné | 0,631 | 15,5 %         | 25,6 %        |
|      1 | XGBoost                            | figé       | 0,607 | 15,5 %         | 5,4 %         |
|      1 | XGBoost                            | réentraîné | 0,607 | 15,5 %         | 5,4 %         |
|      2 | Grille de score (WoE + logistique) | figé       | 0,569 | 7,5 %          | 3,1 %         |
|      2 | Grille de score (WoE + logistique) | réentraîné | 0,589 | 7,5 %          | 3,7 %         |
|      2 | XGBoost                            | figé       | 0,585 | 7,5 %          | 4,0 %         |
|      2 | XGBoost                            | réentraîné | 0,580 | 7,5 %          | 3,0 %         |
|      3 | Grille de score (WoE + logistique) | figé       | 0,628 | 27,5 %         | 9,4 %         |
|      3 | Grille de score (WoE + logistique) | réentraîné | 0,639 | 27,5 %         | 14,7 %        |
|      3 | XGBoost                            | figé       | 0,602 | 27,5 %         | 5,8 %         |
|      3 | XGBoost                            | réentraîné | 0,623 | 27,5 %         | 10,7 %        |
|      4 | Grille de score (WoE + logistique) | figé       | 0,503 | 51,2 %         | 20,1 %        |
|      4 | Grille de score (WoE + logistique) | réentraîné | 0,676 | 51,2 %         | 25,8 %        |
|      4 | XGBoost                            | figé       | 0,535 | 51,2 %         | 7,0 %         |
|      4 | XGBoost                            | réentraîné | 0,625 | 51,2 %         | 16,0 %        |

## Exclure les variables instables ?

Variables au PSI > 0,25 entre les deux moitiés de l'apprentissage : Jours depuis le contact, Contacts antérieurs, Prêt immobilier, Canal de contact, Mois du contact, Résultat campagne précédente. Grille sans ces variables : AUC hors temps 0,612, contre 0,647 pour la grille complète.

## Grille de score

Variables retenues (IV ≥ 0,02, coefficients positifs), par IV décroissante : Mois du contact (0,354), Situation familiale (0,048), Solde moyen (0,047), Profession (0,030), Prêt immobilier (0,027), Âge (0,026), Études (0,026). Retirée pour signe incohérent (colinéarité) : Canal de contact.

Points des trois variables les plus informatives (600 points pour une cote de 1 contre 19, +20 points quand la cote double) :

| variable            | classe                   |   effectif | taux   | woe   |   points |
|:--------------------|:-------------------------|-----------:|:-------|:------|---------:|
| Mois du contact     | août                     |       5215 | 5,5 %  | -0,05 |       85 |
| Mois du contact     | autres modalités (< 3 %) |        539 | 36,4 % | 2,22  |      146 |
| Mois du contact     | février                  |       2296 | 11,1 % | 0,70  |      105 |
| Mois du contact     | janvier                  |       1176 | 3,2 %  | -0,60 |       70 |
| Mois du contact     | juillet                  |       6380 | 6,0 %  | 0,04  |       87 |
| Mois du contact     | juin                     |       4486 | 4,5 %  | -0,27 |       79 |
| Mois du contact     | mai                      |       7957 | 3,2 %  | -0,61 |       70 |
| Mois du contact     | novembre                 |       3598 | 6,1 %  | 0,05  |       88 |
| Situation familiale | divorcé(e)               |       3799 | 7,3 %  | 0,24  |       92 |
| Situation familiale | marié(e)                 |      19782 | 4,9 %  | -0,19 |       82 |
| Situation familiale | célibataire              |       8066 | 7,4 %  | 0,26  |       92 |
| Solde moyen         | ≤ 0                      |       5692 | 4,5 %  | -0,28 |       79 |
| Solde moyen         | ]0 ; 139]                |       4869 | 4,5 %  | -0,26 |       80 |
| Solde moyen         | ]139 ; 397]              |       5263 | 5,4 %  | -0,08 |       84 |
| Solde moyen         | ]397 ; 877]              |       5283 | 5,9 %  | 0,01  |       87 |
| Solde moyen         | ]877 ; 2147]             |       5266 | 6,7 %  | 0,15  |       90 |
| Solde moyen         | > 2147                   |       5274 | 7,9 %  | 0,33  |       95 |
