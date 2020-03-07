
import numpy as np
import matplotlib.pyplot as plt
import gpflow

def dbtime(x):
    return 2*np.sin(x)+2*np.sin(x*2)+5+np.sin(x/2)

class Optimize():

    def __init__(self, func, start_point ):
        self.func = func
        self.start_point = start_point
        self.X = np.random.rand(start_point,1)*10
        self.Y = self.func(self.X)
        self.run = 0
        self.plt = plt

    def buildModelGaussien(self):
        '''
            Réalise la régression gaussienne avec GPFlow
        '''
        # Definition du kernel
        self.k = gpflow.kernels.Matern52(1, lengthscales=0.3)
        # Definition du model
        self.m = gpflow.models.GPR(self.X, self.Y, kern=self.k)
        # Ajustement du lengthscales
        self.m.kern.lengthscales.trainable = False
        self.m.kern.lengthscales = 1.0
        # Compilation du model
        self.m.compile()
        # Optimisation
        gpflow.train.ScipyOptimizer().minimize(self.m)

    def getNextPoint(self):
        '''
            Determine le prochain point à explorer
            Se base sur une focntion d'acquisition: fct = -mean+var
        '''
        # On construit un vecteur de 100 point entre 0 et 10 qui est notre abscisse graphique
        xx = np.linspace(0, 10, 100)[:,None]
        # On réalise toutes les prédictions sur depuis ce vecteur pour obtenir mean et var
        # mean et var sont aussi des vecteurs
        mean, var = self.m.predict_y(xx)
        # Vecteur resultat de la fonction d'acquisition
        # acqu est un matric: une colonne par paramètre
        acqu = (-mean+var)
        # Dans cette exemple il n'y a qu'un paramètre, on met a plat la matrice
        acquflatten = acqu.flatten()
        # On cherche la plus grande valeur
        maxvalue = max(acquflatten)
        # On trouve le paramètre assicié
        whereisit = np.where(acquflatten ==maxvalue )[0][0]
        # Le prochain points d'asbcisse est donc:
        next_abs = xx[whereisit][0]
        # On enrichie X et Y
        self.X = np.append(self.X, [[next_abs]], axis=0)
        self.Y = np.append(self.Y, [[self.func(next_abs)]], axis=0)
        self.run += 1
        return next_abs


    def plot_mg(self):
        xx = np.linspace(0, 10, 100)[:,None]
        mean, var = self.m.predict_y(xx)
        self.plt.plot(self.X, self.Y, 'kx', mew=2)
        self.plt.plot(xx, mean, 'b', lw=2)
        self.plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)

    def plot_acq(self):
        xx = np.linspace(0, 10, 100)[:,None]
        mean, var = self.m.predict_y(xx)
        self.plt.plot(xx, (-mean+var)/5-4, color='green')        

    def print(self):
        print(self.X)
        print(self.Y)

    def savefig(self):
        self.plt.figure(self.run)
        self.plt.grid()
        self.plt.xlim(0,10)
        self.plt.ylim(-6,10)
        self.plt.title("Run %s après un depart aléatoire de %s points" % (self.run, self.start_point))
        self.plot_dbtime()
        self.plot_mg()
        self.plot_acq()
        self.plt.savefig('result6a_%s.png' % (self.run))

    def plot_dbtime(self):
        xfct = np.arange(0,np.pi*4,0.1)
        yfct = self.func(xfct)
        self.plt.plot(xfct,yfct, color='red')

print ('Starting...')
opt = Optimize( dbtime, 5 )
opt.buildModelGaussien()
#opt.print()
opt.savefig()

print ('Searching for min...')
for i in range(20):
    next_abs = opt.getNextPoint()
    print(next_abs)
    opt.buildModelGaussien()
    #opt.print()
    opt.savefig()




