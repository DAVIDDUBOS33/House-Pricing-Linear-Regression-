# Prédire le prix d'une propriété immobilière
### Construction d'un modèle simple de régression linéaire pour commencer en Machine Learning

## Introduction

Si l'on devait demander à des futurs propriétaires ce que serait la maison de leurs rêves, nous aurions surement autant de réponses différentes que de personnes à qui l'on a posé cette question.

Et pourtant, il serait intéressant de pouvoir prédire le prix d'une maison en fonction des de ces critères. C'est ce que nous allons faire dans cet article. Nous disposons d'une base de données qui vient de [Kaggle](https://kaggle.com) du nom de [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

Dans ce dataset, Kaggle a rassemblé 79 variables explicatives du prix d'une maison dans la ville de Ames en Iowa. Nous allons donc construire un modèle de régression linéaire pour tenter de prédire au mieux le prix d'une maison en fonction de ces variables.

## Choix du modèle

Puisque nous cherchons à prédire un chiffre précis (à savoir un prix), il semblerait que la régression soit effectivement le modèle le plus adapté.

Par souci de compréhension et de facilité, nous avons arbitrairement choisi un modèle de régression linéaire multiple mais il existe d'autres modèles voire des mélanges de modèles qui peuvent être plus précis que la régression linéaire comme XGBoost ou Random Forest Regression. Notre régression linéaire va tout de même faire l'affaire et nous vous expliquons pourquoi.

Rappelons tout d'abord les principales hypothèses sur lesquelles se fondent une régression linéaire :​

  * Linéarité : Il faut que votre dataset ait une évolution linéaire
  * Homoscedasticité : La variance de votre dataset ne doit pas être trop forte
  * Non-colinéarité : Il faut que les variables prédictives n'aient pas de relation forte entre elles​

Regardons chacune de ces hypothèses afin de voir pourquoi notre modèle de régression linéaire ne va pas être si mauvais à prédire le prix des maisons. Commencons à coder et analyser notre dataset


### Importer le Training Set et Test Set

Pour cette prédiction, nous aurons besoin des librairies classiques Numpy, Pandas, Matplotlib. Nous importerons scikitlearn plus tard une fois que nous commencerons à nettoyer nos données puis appliquer notre modèle.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

Importons maintenant notre dataset et regardons ce qu'il y a dedans

```python
dataset = pd.read_csv("train.csv")
dataset.head()
```

Nous pouvons déjà constater qu'il y a beaucoup de variables de catégories. Nous allons donc devoir les encoder pour pouvoir les incorporer dans notre modèle. Ne nous attardons pas là dessus tout de suite et regardons tout d'abord les variables numériques.

### Visualisation de quelques variables numériques

Pour voir si notre modèle de régression peut effectivement fonctionner, regardons d'abord la corrélation qu'il y a entre quelques variables numériques et le prix d'une maison.

Commençons par ce qu'il nous parait le plus évident : la taille de la propriété

```python
plt.scatter(dataset["LotArea"], dataset["SalePrice"])
plt.xlabel("Taille de la propriété")
plt.ylabel("Prix de vente")
plt.show()
```

A première vue, hormis quelques données qui sortent du lot, on peut voir que si la taille de la propriété augmente ne serait-ce qu'un peu, le prix de la propriété augmente de beaucoup.

Si l'on enlevait les outliers, on peut voir une certaine linéarité dans la progression. De plus les points ne sont pas si eloignés les uns des autres donc on peut dire qu'il y a une certaine homoscédasticité de la distribution.
Attention cependant, dû à ces outliers, l'échelle de notre graphique est assez grande et nous empêche de voir clairement une relation linéaire. Regardons donc d'autres variables.


```python
plot = dataset.groupby("OverallCond")["SalePrice"].mean()
plt.scatter(np.unique(dataset["OverallCond"]), plot)
plt.xlabel("Condition de la propriété (de 1 à 10)")
plt.ylabel("prix de vente moyen")
plt.show()
```

Ici nous avons calculé le prix de vente moyen d'une propriété en fonction de son état général. Même si les notes 2, 3 et 9 ont l'air anormalement élevées, nous pouvons en tous cas voir que plus la condition générale de la propriété est bonne, plus on va pouvoir la vendre chère.

Regardons une dernière variable.

```python
plot = plot = dataset.groupby("YearBuilt")["SalePrice"].mean()
plt.scatter(np.unique(dataset["YearBuilt"]), plot)
plt.xlabel("Année")
plt.ylabel("prix de vente moyen")
plt.show()
```


Ici, nous pouvons voir le prix de vente moyen d'une maison en fonction de l'année où celle-ci a été construite. Il apparaît assez clair que la tendance est : plus la maison a été construite récemment plus elle va se vendre chère.

D'après les trois variables que nous avons regardées, on peut dire qu'il y a une corrélation qui n'est pas exactement linéaire entre les variables et le prix de vente. Certaines ne le sont pas du tout même.

Cependant, il y a tout de même une certaine homoscédascité puisque, malgré quelques outliers, les points ne sont pas trop éloignés les uns des autres.

Notre modèle de régression linéaire ne va donc pas être parfait mais il va pouvoir déjà nous donner une bonne vision d'ensemble et des prédictions qui ne seront pas si éloignées que ca de la réalité.

En ce qui concerne la non-colinéarité, si nous regardons certaines variables, on peut dire qu'il y a une colinéarité forte. En effet, nous une variable OverallCond qui correspond à la qualité générale de la maison et une variable OverallQual qui correspond à la qualité des matériaux utilisés pour construire la maison. On peut tout à fait se dire que si la qualité générale de la maison est bonne, les materiaux vont encore être de bonne qualité.

De la même manière, en explorant le dataset, vous verrez qu'il y a d'autres variables qui très certainement extrêmement colinéaire.

Nous avons deux possibilités qui s'offrent à nous :

* Enlever les variables qui semblent être colinéaires
* Laisser toutes les variables et construire notre modèle tel quel

Dans le premier cas, l'opération est très longue et nécessite beaucoup de patience. C'est pourquoi nous opterons pour le second cas. En effet, laisser toutes les variables ne va pas détériorer le caractère prédictif de notre modèle. Nous ne pourrons simplement pas savoir quelles variables ont le plus d'influence sur notre modèle mais, puisque nous voulons surtout créer une prédiction, nous nous concentrerons là dessus.


# Construction du modèle
### Détection des valeurs manquantes

Puisque nous sommes sur un dataset assez large, il va être difficile de repérer à vu d'oeil les colonnes où l'on va avoir des valeurs manquantes. C'est pourquoi nous allons construire une fonction qui les détecte et affiche la colonne dans laquelle elle en trouve.

```python
def isnan(dataframe, column):
    for i in range(0, column):
            if dataframe.iloc[:,i].isnull().any() == True:
                print("Column ", i, "has Nan")


isnan(dataset, 81)
```

Ca nous fait un bon paquet de colonnes à s'occuper ! Voyons comment on peut procéder

### Séparation des variables indépendantes de la variable dépendantes

Commençons tout d'abord par séparer nos variables dépendantes de notre variable indépendante. Il sera plus simple de s'y retrouver ensuite

```python
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 80:81].values
```

Nous prenons ici toutes les variables de notre dataset sauf la dernière puisque c'est le prix de notre maison et c'est ce que nous voulons prédire.

### Remplacer les valeurs manquantes dans les variables numériques

Lorsque nous regardons notre matrice X, on peut voir des valeurs manquantes dans la colonne 2, 25 et 58 qui correspondent respectivement aux colonnes MSSubClass (type d'habitation) au MasVnrArea (surface en pied du type de maconnerie) et GarageYrBlt (l'année où le garage a été construit).

Nous allons remplacer les valeurs manquantes par la valeur médiane pour ne pas biaiser le reste du dataset. Cependant la médiane risque de biaiser la colonne GarageYrBlt puisque lorsque l'on a une valeur manquante cela veut simplement dire qu'il n'y a pas de garage dans la maison. Restons simple pour l'instant et gardons la médiane mais gardons cela en tête comme piste d'amélioration de notre modèle.

```python
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "median", axis = 0)
imputer.fit(X[:, [2, 25, 58]])
X[:,[2, 25, 58]] = imputer.transform(X[:, [2, 25, 58]])
```

### Remplacer les valeurs manquantes dans les variables texte

Dernière étape est de remplacer les valeurs manquantes dans les colonnes textes. Cette étape sera très simple. Il suffit de transformer notre X en un DataFrame via Panda puis utiliser la fonction `fillna()` pour remplacer les NaN par une valeur texte.

Lorsque nous regardons ce à quoi correspond les NaN dans les colonnes texte, elles caractérisent l'absence de la catégorie dans la propriété. Par exemple, les NaN dans la colonne GarageFinish correspond au fait qu'il n'y a pas de garage dans la maison.

Dès lors, nous allons simplement remplacer les valeurs manquantes par : `None` et le tour est joué.


```python
X = pd.DataFrame(X)
X = X.fillna("None")
X.head()
```
### Encoder les variables catégoriques

Dernière étape avant de finir notre phase de preprocessing est d'encoder nos variables catégoriques.

Nous rencontrons ici un second problème. Notre matrice X est composée de variables catégoriques nominales et ordinales. Techniquement, nous devrions simplement encoder les variables ordinales en chiffres et ne pas les traiter comme des variables factices de façon à respecter "l'ordre" des catégories.

Cependant, nous avons 43 variables textes dont 21 qui sont ordinales et 22 qui ne sont nominales. Deux options s'offrent encore à nous :

1. Encoder les variables ordinales à la main puis les variables nominales automatiquement
2. Traiter toutes les variables comme des variables nominales et les encoder rapidement avec quelques lignes de code


Par soucis de rapidité, nous choisirons l'option 2. Certes, elle ne sera pas aussi précise que l'option 1 et nous allons nous retrouver avec une nouvelle matrice X composée de beaucoup plus de colonnes. Mais puisque nous avons décidé auparavant que nous ne chercheront pas à optimiser notre modèle, nous n'effectueront pas de backward elimination pour traiter chacune de nos variables.
De plus la qualité prédictive de notre modèle n'en sera pas trop amoindrie si nous encodons nos variables ordinales comme des variables nominales.

```python
X = X.values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
k = []
for i in range(0,79):
    if type(X[0,i]) == str:        
        X[:,i] = labelencoder.fit_transform(X[:,i])
        k +=[i]

onehotencoder = OneHotEncoder(categorical_features = [k])    
X = onehotencoder.fit_transform(X).toarray()
```

NB : N'oublions pas de reconvertir X en une array (X = X.values) sinon le code du dessous ne fonctionnera pas.

### Application de notre modèle de régression linéaire

Notre phase de data preprocessing est maintenant terminée. Nous pouvons donc séparer notre dataset en training set et un test set. Nous choisirons un ration de 80 / 20 pour séparer nos données.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
```
Appliquons maintenant notre modèle de régression linéaire sur notre training set et faisons notre première prédiction sur X_test.


```python
from sklearn.linear_model import LinearRegression
regressor_lr = LinearRegression()
regressor_lr.fit(X_train, y_train)

y_pred_lr = regressor_lr.predict(X_test)
```

Tout fonctionne bien. Regardons les premières prédictions de notre modèle par rapport à nos valeurs test

```python
overview_y_pred = y_pred_lr
overview_y_test = y_test

overview = pd.DataFrame(data=np.column_stack((overview_y_pred,overview_y_test)),
                        columns=['Predictions','Valeurs Réelles'])

overview.head()
```

# Evaluation de notre modèle

Notre modèle est construit et semble être capable de prédire de manière assez précise la valeur d'une propriété à quelques milliers d'euros près. Cependant, nous l'avons testé que sur les 5 premières valeurs.

Une manière assez simple d'évaluer notre modèle serait de voir l'écart moyen entre nos valeurs prédites et nos valeurs réelles. De cette manière, nous aurons une vue globale de la performance de notre modèle.

```python
accuracy_lr = []

for i in range (0, 292):
    if y_test[i] - y_pred_lr[i] < 0:
        accuracy_lr.append(y_pred_lr[i] - y_test[i])
    else:
        accuracy_lr.append(y_test[i] - y_pred_lr[i])


accuracy_lr = np.asarray(accuracy_lr)
accuracy_lr.mean()
```

Nous pouvons voir ici que nous avons un écart moyen entre nos valeurs réelles et nos valeurs prédites de 16 624$. Ce qui correspond à 10% d'écart environs. Ce qui est vraiment bien pour un modèle de régression linéaire simple !

# Pistes d'amélioration du modèle

Il est toujours possible d'améliorer un modèle. Voici quelques idées qui pourraient faire la différence :

* Regarder la co-linéarité entre les variables et retirer celles qui une relation trop forte
* Encoder les variables ordinales différement des variables nominales
* Améliorer la façon dont nous avons traité les valeurs manquantes

Dans le premier cas, il se peut en effet qu'enlever des variables qui ont une forte co-linéarité rende le modèle plus performant. Vous aurez aussi moins de variables à étudier donc le modèle sera plus facile à traiter.
Dans le second cas, nous aurions simplement dû appliquer LabelEncoder() sur nos variables ordinales de manière à garder l'influence de l'ordre des catégories dans notre modèle.

Enfin nous avons remplacé toutes les valeurs manquantes dans les variables numériques par la médiane. Cependant, la colonne GarageYrBlt correspond à l'année où le garage a été construit. Or, nous savons que dans certaines maisons, il n'y a pas eu de garage construit. Au lieu de remplacer donc les dates manquantes par la médiane, peut être serait-il judicieux de remplacer par une autre valeur comme 9999 ou une année future dans le temps.


Si vous êtes intéressé à l'idée d'apprendre les Data Sciences, regardez notre site : [Jedha.co](https://jedha.co)
