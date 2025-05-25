# Origine de la Binary Cross-Entropy function (aussi appellee log loss):

## formule:

E = -1/N * (Somme de n = 1 a N de ( yn * log(pn) + (1 - yn) * log(1 - pn) )) 
Ou N est la taille du data set, yn est le resultat attendu (0 ou 1), et pn est la probabilitee calculee par le modele

Le but est de calculer les pertes du modele en calculant la difference entre les vraies valeures et celles obtenues pas le modele
Pour pouvoir les minimiser via une desscente de gradient.

Mais d'ou vient cette formule ?

En realite, elle est assez simple a comprendre:

On cherche a la base a calculer la vraissemblence de tous les resultats, c'est a dire faire le produit de toutes les probabilitee
Nos resultats seront 0 ou 1, on obtient donc ces probabilitees:

P(Y = 1) = a (la probabilitee calculee pas la fonction d'activation, on utilisera la fonction Sigmoïde)
Et reciproquement:
P(Y = 0) = 1 - a

On a :
$$
a = \frac {1} {1 + e^{-z}}
$$
Et
$$
z = w_nx_n + b
$$

On peut ensuite resumer le tout en une seule formule:
P(Y = y) = (a(z) ** y) * ((1 - a(z)) ** 1 - y)
(exemple: P(Y = 1) = (a(z) ** 1) * ((1 - a(z)) ** 0) <=> P(Y = y) = (a(z)) * 1)

on remplace a(z) par p pour la lisibilite (p = proba renvoyee pas a(z))
Ensuite, pour multiplier toutes les probabilitees, plus qu'a faire:
L(Likelihood) = Produit de n = 1 a N de (pn ** yn) * ((1 - pn) ** 1 - yn)

Cependant, en faisant un produit de probabilitees, on va se rapprocher de plus en plus de 0 (cat P toujours <= 1),
ce qui peut poser probleme a l'ordinateur pour les calculs au bout d'un moment.

C'est la qu'intervient log(), car log(a * b) = log(a) + log(b) et que log() est une fonction monotonne croissante, 
elle conserve donc l'ordre des thermes.

On peut donc faire cela:
LL = log(L) = log(Produit de n = 1 a N de (pn ** yn) * ((1 - pn) ** 1 - yn))
On ferait donc (log(0.8 * 0.4 * 0.7) = log(0.8) + log(0.4) + log(0.7) + log(0.1) = -1.6498  plutot que 0.8 * 0.4 * 0.7 * 0.1 = 0.0224)

Puisque la fonction log conserve l'ordre des thermes, chercher le max du log de la vraissemblence revient a chercher le 
max de la vraissemblence

Plus qu'a transformer un peu cette formule:

log(Produit de n = 1 a N de (pn ** yn) * ((1 - pn) ** 1 - yn))
<=> Somme de n = 1 a N de log((pn ** yn) * ((1 - pn) ** 1 - yn)) (car logarithme d'un produit equivaut a la somme des logarithmes)
<=> Somme de n = 1 a N de log(pn ** yn) + log((1 - pn) ** 1 - yn) (car encore un produit)
<=> Somme de n = 1 a N de yn * log(pn) + 1 - yn * log((1 - pn)) (car log(x**n) = nlog(x))

Finalement pour arriver a la fonction E, il ne manque que -1/N devant,
Ici on cherchais a faire une maximisation de la vraissemblence, mais en realite, il est plus approprie de minimiser une fonction
on va alors minimiser la fonction inverse qui est logiquement:
-1 * (Somme de n = 1 a N de yn * log(pn) + 1 - yn * log((1 - pn)) (car log(x**n) = nlog(x)))

Puis on divise finalement le tout par N pour normaliser le resultat final, on obtient donc bien:
E = -1/N * (Somme de n = 1 a N de ( yn * log(pn) + (1 - yn) * log(1 - pn) ))

$$
E = -\frac{1}{N} \sum_{n=1}^{N} \left[ y_n \log(p_n) + (1 - y_n) \log(1 - p_n) \right]
$$


#
# Calcul des differentes derivees

On va ensuite calculer les differentes derivees partielles afin de faire ensuite une descente de gradiant, comme dans la regression lineaire, pour minimiser le cout de la fonction log loss

## Derivee de LogLoss par wn

La derivee de L par w1 est egale a la derivee de L par rapport a a (a = fonction d'activation = p(n)), le tout multiplie par la derivee de a par z, et finalement le tout multiplie par la derivee de z par w1

Cela donne:

$$
\frac{\partial \mathcal{L}}{\partial \omega_n} 
= 
\frac{\partial \mathcal{L}}{\partial a} 
\times 
\frac{\partial a}{\partial z} 
\times 
\frac{\partial z}{\partial \omega_n}
$$

On commence par calculer 

$$\frac{\partial \mathcal{L}}{\partial a}$$

On recupere
$$
E = -\frac{1}{N} \sum_{n=1}^{N} \left[ y_n \log(p_n) + (1 - y_n) \log(1 - p_n) \right]
$$

On remplace p par a pour que ca soit plus clair:

$$
-\frac{1}{N} \sum_{n=1}^{N} \left[ y_n \log(a_n) + (1 - y_n) \log(1 - a_n) \right]
$$

Les derivees de 
$$
log(a) 
$$
$$
log(1 - a)
$$
sont
$$
\frac {1} {a}
$$

$$
\frac {-1} {1 - a}
$$
car les derivees de log(x) et log(u) sont
$$
\frac {1} {x}
$$
$$
\frac {1} {u}
\times
u'
$$


$$
\frac {\partial \mathcal{L}}{\partial a} = -\frac{1}{N} \sum_{n=1}^{N} \left[ y \frac {1}{a} + (1 - y) \frac {-1}{1-a} \right]
\\
<=>
\frac {\partial \mathcal{L}}{\partial a} = -\frac{1}{N} \sum_{n=1}^{N} \left[\frac {y}{a} -  \frac {1 - y}{1-a} \right]
$$

Ensuite, derivee de 

$$\frac{\partial \mathcal{a}}{\partial z}$$

On a 
$$
a(z) = \frac {1} {1 + e^{-z}}
$$

on peut decomposer sa derivee en
$$
g = \frac {1}{f}
$$
$$
f = 1 + e^{-z}
$$

Donc 
$$
\frac{\partial \mathcal{a}}{\partial z} = g'(f(z)) \times f'(z)
$$

Or 

$$
f' = -e^{-z}
$$
$$
g' = \frac {-1}{f^2}
\\<=>
g' = \frac {-1}{(1 + e^{-z})^2}
$$

Donc

$$
\frac{\partial \mathcal{a}}{\partial z} = \frac {-1}{(1 + e^{-z})^2} \times (-e^{-z})
\\<=>
\frac{\partial \mathcal{a}}{\partial z} = \frac {e^{-z}}{(1 + e^{-z})^2}
\\<=>
\frac{\partial \mathcal{a}}{\partial z} = \frac {1}{1 + e^{-z}}
\times \frac {e^{-z}}{1 + e^{-z}}
\\<=>
\frac{\partial \mathcal{a}}{\partial z} = a \times \frac {e^{-z} + 1 - 1}{1 + e^{-z}}
\\<=>
\frac{\partial \mathcal{a}}{\partial z} = a \times \frac {e^{-z} + 1}{1 + e^{-z}}
+ \frac {- 1}{1 + e^{-z}}
\\<=>
\frac{\partial \mathcal{a}}{\partial z} = a \times (1 - a)
$$

Finalement, on calcule 
$$
\frac{\partial z}{\partial \omega_n}
$$

On a 
$$
z(w_n) = w_nx_n+b
$$

Donc
$$
z'(w_n) = x_n
$$

Recapitulons, nous avons:

$$
\frac {\partial \mathcal{L}}{\partial a} = -\frac{1}{N} \sum_{n=1}^{N} \left[\frac {y}{a} -  \frac {1 - y}{1-a} \right]
$$
$$
\frac{\partial \mathcal{a}}{\partial z} = a \times (1 - a)
$$
$$
\frac{\partial z}{\partial \omega_n} = x_n
$$
On cherchais 

$$
\frac{\partial \mathcal{L}}{\partial \omega_n} 
= 
\frac{\partial \mathcal{L}}{\partial a} 
\times 
\frac{\partial a}{\partial z} 
\times 
\frac{\partial z}{\partial \omega_n}
$$

On a donc finalement

$$
\frac{\partial \mathcal{L}}{\partial \omega_n} = 
(-\frac{1}{N} \sum_{i=1}^{N} \left[\frac {y_i}{a_i} -  \frac {1 - y_i}{1-a_i} \right])
\times a_i \times (1 - a_i)
\times x_n
\\ <=>
\frac{\partial \mathcal{L}}{\partial \omega_n} = 
-\frac{1}{N} \sum_{i=1}^{N} \left[y_i(1-a_i) - (1 - y_i)a_i \right]
\times x_n
\\ <=>
\frac{\partial \mathcal{L}}{\partial \omega_n} = 
-\frac{1}{N} \sum_{i=1}^{N} \left[y_i-y_ia_i - a_i + y_ia_i \right]
\times x_n
\\ <=>
\frac{\partial \mathcal{L}}{\partial \omega_n} = 
-\frac{1}{N} \sum_{i=1}^{N} \left[y_i - a_i \right]
\times x_n
$$

La deuxieme derivee necessaire pour la desscente de gradient est celle par rapport au bias

$$
\frac{\partial \mathcal{L}}{\partial b_n}
$$

On derive donc cela par rapport a b
$$
-\frac{1}{N} \sum_{n=1}^{N} \left[ y_n \log(a_n) + (1 - y_n) \log(1 - a_n) \right]
$$

En suivant la meme logique que precedamment, on a les memes derivees, exceptee la derniere qui devient:

$$
z'(w_n) = 1
$$
car
$$
z = w_nx_n + b
$$

On a donc finalement
$$
\frac{\partial \mathcal{L}}{\partial b_n} = 
-\frac{1}{N} \sum_{i=1}^{N} \left[y_i - a_i \right]
\times 1
\\ <=>
\frac{\partial \mathcal{L}}{\partial b_n} = 
-\frac{1}{N} \sum_{i=1}^{N} \left[y_i - a_i \right]
$$

On a maintenant les 2 derivees necessaires a la desscente de gradient.