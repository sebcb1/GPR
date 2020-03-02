
import numpy as np
import matplotlib.pyplot as plt
import gpflow

def dbtime(x):
    return 2*np.sin(x)+2*np.sin(x*2)+5+np.sin(x/2)

def plot_mg(m):
    xx = np.linspace(0, 10, 100)[:,None]
    mean, var = m.predict_y(xx)
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)

def max_acq(m):
	xx = np.linspace(0, 10, 100)[:,None]
	mean, var = m.predict_y(xx)
	acqu = (-mean+var)
	acquflatten = acqu.flatten()
	maxvalue = max(acquflatten)
	whereisit = np.where(acquflatten ==maxvalue )[0][0]
	return xx[whereisit][0]

def plot_acq(m):
	xx = np.linspace(0, 10, 100)[:,None]
	mean, var = m.predict_y(xx)
	plt.plot(xx, (-mean+var)/5-4, color='green')

def plot_dbtime():
	xdbtime = np.arange(0,np.pi*4,0.1)
	ydbtime = dbtime(xdbtime)
	plt.plot(xdbtime,ydbtime, color='red')

def modelGaussien(X, Y):
	k = gpflow.kernels.Matern52(1, lengthscales=0.3)
	m = gpflow.models.GPR(X, Y, kern=k)
	m.kern.lengthscales.trainable = False
	m.kern.lengthscales = 1.0
	m.compile()
	gpflow.train.ScipyOptimizer().minimize(m)
	return m


# Je prende des points X au hazard et je détermine Y
N = 5
X = np.random.rand(N,1)*10
Y = dbtime(X)

# Je construit mon modèle
m = modelGaussien(X, Y)

# Configuration de mon graph
plt.figure(0)
plt.grid()
plt.xlim(0,10)
plt.ylim(-6,10)
plt.title("Fonction dbtime, régression et acquisition run %s" % (0))

# Ajout des fonctions
plot_dbtime()
plot_mg(m)
plot_acq(m)
plt.savefig('run%s.png' % (0))

for i in range(1,5):
		next_pointx = max_acq(m)
		print('%s %s' % (next_pointx, dbtime(next_pointx)))
		X = np.append(X, [[next_pointx]], axis=0)
		Y = np.append(Y, [[dbtime(next_pointx)]], axis=0)
		print(X)
		print(Y)
		m = modelGaussien(X, Y)
		plt.figure(i)
		plt.grid()
		plt.xlim(0,10)
		plt.ylim(-6,10)
		plt.title("Fonction dbtime, régression et acquisition run %s" % (i))
		plot_dbtime()
		plot_mg(m)
		plot_acq(m)
		plt.savefig('run%s.png' % (i))

