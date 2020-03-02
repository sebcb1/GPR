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
```

## Mise en place initiale de l'environement virtuel

```
pipenv --python /usr/bin/python3
pipenv install gpflow
pipenv install numpy
```

## Les exemples

### Exemple 1: Une fonction sigmoide 

Pour simuler l'effet d'un cache sur le temps de réponse

```
pipenv run python run1.py
```

![Cache Type](GPR/exemple1/cache_type.png)

## Outils et sources

* [Editeur Sublim](https://www.sublimetext.com/)
* [Installation du plugins Markdown Preview](http://plaintext-productivity.net/2-04-how-to-set-up-sublime-text-for-markdown-editing.html)
* [Documentations GPFlow](https://gpflow.readthedocs.io/en/stable/index.html)




