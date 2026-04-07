# guided-tour-ml-in-finance
Repo du MOOC "Guided Tour of Machine Learning In Finance"

## Module 2 : Mathematical foundations of machine learning
### Lab : `Euclidian_distance.ipynb`
Ce TP a deux buts :
- nous montrer l'importance du calcul vectorisé, pour rendre efficace les opérations qui manipulent des vecteurs ou des matrices. De telles opérations permettent d'utiliser le parallélisme de nos processeurs (on remplace les boucles Python for par des opérations optimisées)
Avec la méthode de calcul à la main (avec une boucle for donc), j'ai même du stopper le calcul des distances des 10 000 points tellement il prenait de temps sur mon ordinateur. Je suis passé à 1000 : il s'effectue alors en 6 secondes environ. La version vectorisée, elle, en 0.1 seconde environ.

- un résultat plus théorique : curse of dimensionnality. On regarde comment évolue la distance de points de dimension N générés aléatoirement, à mesure que N augmente.
On observe empiriquement que, à mesure que N augmente :
  - la distance moyenne augmente (en racine de N)
  - la variance de la distance est stable
  - l'assymétrie de la distribution des distances tend vers 0
  - la kurtosis tend vers 0, c'est-à-dire que la distribution se rapporche de plus en plus d'une gaussienne
 
Distribution des distances entre 10000 points de dimension N=2 générés aléatoirement :
![n2](https://github.com/user-attachments/assets/dde11a00-963d-4e7a-8aa1-0d0fe7ca7878)

Idem mais N=10 :

![n10](https://github.com/user-attachments/assets/391b4a69-0d3c-4f68-9d55-7be6a07e9287)

Evolution de la kurtosis à mesure que N augmente :

![kurtosis](https://github.com/user-attachments/assets/61e81564-49e0-46b8-a5f0-0407b5d5887e)

(d'autres graphiques sont visibles dans le fichier du tp)
C'est le phénomène de curse of dimensionality (https://en.wikipedia.org/wiki/Curse_of_dimensionality#Distance_function) : la fonction distance perd son intérêt en haute dimension puisqu'elle devient de moins en moins "discriminante" (donc les aglos comme nearest neighbor deviennent moins pertinents)

## Module 3 : Mathematical foundations of machine learning
### Lab : `Linear_regression.ipynb`

Le but du TP est de se familiariser avec TensorFlow et pour cela d'implémenter une régression linéaire assez simple pour comprendre comment la librairie fonctionne. Pour cela, on implémente à la fois la méthode avec l'équation normale mais aussi celle avec descente de gradient.

Graphiques issus du TP (non demandés dans le TP mais que j'ai souhaité visualisé en plus) :
Visualiser des données de test (donc jamais vues pendant l'entraînement) ainsi que la prédiction du modèle en rouge :
<p>
  <img width="421" height="385" alt="Capture d’écran 2026-04-03 à 18 56 49" src="https://github.com/user-attachments/assets/a334149e-3908-4f44-b78d-579576023b9a" />
</p>

On voit globalement que le modèle est assez fidèle et donc qu'il a une erreur de généralisation faible.

Visualisation de l'évolution du coût pendant l'entraînement (méthode descente de gradient) :
<p>
<img width="435" height="277" alt="Capture d’écran 2026-04-03 à 18 56 57" src="https://github.com/user-attachments/assets/4764f8d0-7ec2-4dc2-971e-0b7e5794a68e" />
</p>
