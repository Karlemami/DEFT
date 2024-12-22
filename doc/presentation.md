# Tâches
3 tâches :
– détection du caractère objectif ou subjectif global d’un texte depuis un corpus d’articles de journaux ;
– détection des passages subjectifs d’un texte, sur deux corpus : articles de journaux et débats parlementaires ;
– détermination du parti politique d’appartenance de chaque intervenant dans le corpus parlementaire

# Corpus
## Articles de journaux
– Le corpus en français est issu du quotidien Le Monde, 42 000 articles sur les années 2003 à 2006 ;
– Le corpus en anglais provient du quotidien économique The Financial Times, 13 000 articles de l’année 1993 ;
– Le corpus en italien comprend 2 500 articles du journal économique Il Sole 24 Ore sur la période 1992/1993.

## Débats parlementaires

archives multilingues de 313 séances parlementaires tenues entre 1999 et 2004. Méta-données : nom du parlementaire, langue dans laquelle il s'exprime, nom du parti européen auquel il appartient (Il existe neuf groupes politiques européens : EDD (Europe des Démocraties et des Différences), ELDR (parti Européen des Libéraux,
Démocrates et Réformateurs), GUE/NGL (groupe confédéral de la Gauche Unitaire Européenne et Gauche Verte Nordique), NI (les non
inscrits), PPE-DE (Parti Populaire Européen (démocrates chrétiens) et Démocrates Européens), PSE (Parti Socialiste Européen), TDI (groupe
Technique des Députés Indépendants), UEN (Union pour l’Europe des Nations) et enfin, les Verts/ALE (Verts, Alliance Libre Européenne).)

# Référence

## Tâche 1
### Le monde
Le secteur de rédaction définit la subjectivité :
- France et International = objectif
- Éditorial-analyses et Débats-décryptages = subjectif

### Financial Times
-"Comment & Analysis" = subjectif
- "General News" = objectif

### Il Sole 24 Ore
-"Opinioni e commentati" = subjectif
-tout le reste = objectif (wtf)

## Tâche 2
Référence créée a posteriori, à partir d'un vote majoritaire entre les résultats des participants.
Seulement deux équipes ont participé à cette tâche. La taille des passages a dû être harmonisée pour le vote majoritaire : la première équipe définissait un passage comme une phrase, la deuxième non. Les tags de la deuxième équipe ont été étendues aux phrases (en gros il faut travailler sur les phrases)

## Tâche 3
Le corpus a été aligné avant le split train/test, donc les interventions utilisées pour l'apprentissage dans une langue ne peuvent pas être présentes dans le test d'une autre langue. Après le split, les corpus ont été shuffle, donc c'est plus parallèle(pour éviter par exemple de travailler sur une seule langue et de dupliquer les résultats pour une langue sur les deux autres).
Seuls cinq partis sont présents dans les données :
- ELDR (3346)
- GUE/NGL (4482)
- PPE-DE (11429)
- PSE (9066)
- ALE (3961)

Note : les résultats pour cette tâche sont très mauvais, pour les humains mais encore plus pour les modèles (3 participants, meilleur score à 0.33 de f-mesure)

# Mesures d'évaluation

Macro-moyenne de f-mesure sur les différentes classes, donc chaque classe compte à égalité avec les autres
