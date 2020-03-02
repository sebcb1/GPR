
import numpy as np
import matplotlib.pyplot as plt
import gpflow

def dbtime(x):
    return 2*np.sin(x)+2*np.sin(x*2)+5+np.sin(x/2)

def plot(m):
    xx = np.linspace(0, 10, 100)[:,None]
    mean, var = m.predict_y(xx)
    #plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)




# Ma fonction dbtime calculé
xdbtime = np.arange(0,np.pi*4,0.1)
ydbtime = dbtime(xdbtime)

# Je prende des points X au hazard et je détermine Y
N = 10
X = np.random.rand(N,1)*10
Y = dbtime(X)

# Je construit mon modèle
k = gpflow.kernels.Matern52(1, lengthscales=0.3)
m = gpflow.models.GPR(X, Y, kern=k)
m.likelihood.variance = 0.01
m.kern.lengthscales.trainable = False
m.kern.lengthscales = 1.0
m.compile()
print(m)
gpflow.train.ScipyOptimizer().minimize(m)
print(m)

# Configuration de mon graph
plt.grid()
plt.xlim(0,10)
plt.ylim(0,10)
plt.title("Fonction dbtime et régression sur 30 points")

# Ajout de la fonction dbtime
plt.plot(xdbtime,ydbtime, color='red')

# Ajout de la fonction régréssé
plot(m)

plt.savefig('dbtime2.png')
