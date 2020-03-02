# Exemple de Processus Gaussien

## Mise en place du serveur

```
- Centos 7.1 Installation
	- Language: English (United States)
	- Software selection: Gnome Desktop
	- KDUMP: disable
	- Security policy: No profile selected
```

## Configuration OS

```
yum update -y

/etc/selinux/config:
SELINUX=disabled

systemctl stop firewalld.service
systemctl disable firewalld.service

yum install -y https://download.postgresql.org/pub/repos/yum/11/redhat/rhel-7-x86_64/pgdg-centos11-11-2.noarch.rpm
yum install python3.x86_64  git.x86_64  gcc.x86_64 python3-devel.x86_64 
yum install postgresql11-server postgresql11 postgresql11-devel.x86_64
pip3 install pipenv
```

## Mise en place depuis github

```
pipenv install
```

## Mise en place initiale de l'environement virtuel

Si on ne passe par par github et Pipfile:

pip3 install --user tensorflow==1.14.0


```
pipenv --python /usr/bin/python3
pipenv install 'tensorflow<2.0'
pipenv install numpy
pipenv install 'gpflow<2.0'
```

## Les exemples

### Exemple 1: Une fonction sigmoide 

Pour simuler l'effet d'un cache sur le temps de réponse

```
pipenv run python run1.py
```

![Cache Type](exemple1/cache_type.png)

### Exemple 2: Une première régression

Une première regression gaussienne à partir d'un ensemble de point

```
pipenv run python run2.py
```

![Cache Type](exemple2/first_regression.png)

### Exemple 3: Régression sur la fonction sigmoide

```
pipenv run python run3.py
```

On commence sans optimisation des hyperparametres (ligne 24 et 25 commentées):

```
#gpflow.train.ScipyOptimizer().minimize(m)
#print(m)
```

On a:
```
                             class prior transform  trainable shape  fixed_shape value
GPR/kern/lengthscales    Parameter  None       +ve       True    ()         True   0.3
GPR/kern/variance        Parameter  None       +ve       True    ()         True   1.0
GPR/likelihood/variance  Parameter  None       +ve       True    ()         True  0.01
```

![step1](exemple3/sigmoide_step1.png)

On peut augmenter le nombrede point à 20 (ligne 12):

![step2](exemple3/sigmoide_step2.png)

Avec optimisation:
```
                             class prior transform  trainable shape  fixed_shape                  value
GPR/kern/lengthscales    Parameter  None       +ve       True    ()         True      5.474471159944401
GPR/kern/variance        Parameter  None       +ve       True    ()         True     1.6626323263700333
GPR/likelihood/variance  Parameter  None       +ve       True    ()         True  1.000001721511822e-06
```

![step3](exemple3/sigmoide_step3.png)

Sur cette dernière images il y a peu de variance entre le 2e et 3e point
Si je veux une confiance plus faible je dois diminuer GPR/kern/lengthscales
Je vais donc le fixé pour diminuer la confiance entre des intervalles trop grandes.

On décommente ligne 20 et 21: 

```
m.kern.lengthscales.trainable = False
m.kern.lengthscales = 1.0
```
![step4](exemple3/sigmoide_step4.png)
 
### Exemple 4: Une fonction dbtime avec un paramêtre

```
pipenv run python run4a.py
```

Ma fonction dbtime sera de la forme suivante.
On notera quelle possède deux minimums locales en plus du minum global.

![step4](exemple4/dbtime1.png)

```
pipenv run python run4b.py
```

Tentative de régression avec 10 points. On notera les zones inconnu avec un potentiel minimum globale

![step4](exemple4/dbtime2.png)

SI on augmente le nombre de point:

![step4](exemple4/dbtime3.png)

![step4](exemple4/dbtime4.png)

A la fin on note que la courbe obtenue colle assez bien à la focntion d'origine
Mais nous avons 30 mesures !!

## Outils et sources

* [Editeur Sublim](https://www.sublimetext.com/)
* [Installation du plugins Markdown Preview](http://plaintext-productivity.net/2-04-how-to-set-up-sublime-text-for-markdown-editing.html)
* [Documentations GPFlow](https://gpflow.readthedocs.io/en/stable/index.html)
* [GitHub GPflow](https://github.com/GPflow/GPflow)
* [Les releases TensorFlow](https://pypi.org/project/tensorflow/#history)




