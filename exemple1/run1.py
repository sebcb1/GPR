import matplotlib.pyplot as plt
import numpy as np

def sigmoide_cache(x):
    return 1.5 /( 1 + np.exp(-(2*x)+2)) - 0.5

x = np.arange(-5,10,0.1)
y2 = sigmoide_cache(x)

plt.grid()
plt.xlim(0,6)
plt.ylim(-0.5,2)
plt.title("Fonction Cache type")

plt.plot(x,y2)

plt.savefig('cache_type.png')
#plt.show()
