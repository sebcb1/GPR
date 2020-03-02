import matplotlib.pyplot as plt
import numpy as np
import gpflow


def sigmoide_cache(x):
    return 1.5 /( 1 + np.exp(-(2*x)+2)) - 0.5

plt.style.use('ggplot')

N = 20
X = np.random.rand(N,1)*6
print(X)
Y = sigmoide_cache(X)

k = gpflow.kernels.Matern52(1, lengthscales=0.3)
m = gpflow.models.GPR(X, Y, kern=k)
m.likelihood.variance = 0.01
m.kern.lengthscales.trainable = False
m.kern.lengthscales = 1.0
m.compile()
print(m)
gpflow.train.ScipyOptimizer().minimize(m)
print(m)

def plot(m):
    xx = np.linspace(0, 6, 100)[:,None]
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
    plt.xlim(0, 6)

plot(m)

plt.savefig('sigmoide_step4.png')