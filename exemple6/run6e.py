
import numpy as np
import matplotlib.pyplot as plt
import gpflow

def dbtime(X):
    x1 = X[:,0]
    x2 = X[:,1]
    return (x1/2-2)*(x1/2-2)+2 + 2*np.sin(x2)+2*np.sin(x2*2)+5+np.sin(x2/2)+2*np.sin(x2)+2*np.sin(x2*2)+5+np.sin(x2/2)

class Optimize():

    def __init__(self, func, start_point, nb_param ):
        self.func = func
        self.nb_param= nb_param
        self.start_point = start_point
        self.X = np.random.rand(start_point,1)*10
        print("Random first parameter:")
        print(self.X)
        for i in range(self.nb_param-1):
            print("Random next parameter:")
            xp = np.random.rand(start_point,1)*10
            self.X = np.concatenate( (self.X, xp),  axis=1 )
        self.Y = self.func(self.X)
        self.Y = self.Y.reshape(len(self.Y), 1)
        self.run = 0
        self.plt = plt

    def buildModelGaussien(self):
        '''
            Réalise la régression gaussienne avec GPFlow
        '''
        # Definition du kernel
        k1 = gpflow.kernels.Matern52(1, active_dims=[0], lengthscales=0.3)
        k1.lengthscales.trainable = False
        k1.lengthscales = 1.0
        k2 = gpflow.kernels.Matern52(1, active_dims=[1], lengthscales=0.3)
        k2.lengthscales.trainable = False
        k2.lengthscales = 1.0        
        self.k = k1 + k2
        # Definition du model
        self.m = gpflow.models.GPR(self.X, self.Y, kern=self.k)
        # Ajustement du lengthscales
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
        xx = None
        for xx1 in range(0,100):
            for xx2 in range(0,100):
                if xx is None:
                    xx = np.array([[xx1,xx2]])
                else:
                    xx = np.concatenate((xx, [[xx1,xx2]]))
        xx = xx / 10
        # On réalise toutes les prédictions sur depuis ce vecteur pour obtenir mean et var
        # mean et var sont aussi des vecteurs
        mean, var = self.m.predict_y(xx)
        #print("variance %s:" % (var))
        # Vecteur resultat de la fonction d'acquisition
        # acqu est un matric: une colonne par paramètre
        acqu = (-mean+var)
        # Dans cette exemple il n'y a qu'un paramètre, on met a plat la matrice
        acquflatten = acqu.flatten()
        # On cherche la plus grande valeur
        maxvalue = max(acquflatten)
        #print("Max found: %s " % maxvalue)
        # On trouve le paramètre assicié
        whereisit = np.where(acquflatten ==maxvalue )[0][0]
        # Le prochain points d'asbcisse est donc:
        next_abs = xx[whereisit]
        # On enrichie X et Y
        self.X = np.concatenate( (self.X, [next_abs]))
        result = self.func(next_abs.reshape((1,2)))
        #print("After search %s: " % result)
        self.Y = np.concatenate( (self.Y, result.reshape((1,1)) ) )
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
        print("Content of X:")
        print(self.X)
        print("Content of Y:")
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
opt = Optimize( dbtime, 10, 2 )
opt.print()
opt.buildModelGaussien()
next_abs = opt.getNextPoint()


print ('Searching for min...')
for i in range(20):
    next_abs = opt.getNextPoint()
    print("Next point to explore: %s and dbtime(x)=%s" % (next_abs,dbtime(next_abs.reshape((1,2)))))
    opt.buildModelGaussien()






