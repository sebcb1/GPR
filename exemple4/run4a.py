
import numpy as np
import matplotlib.pyplot as plt

def dbtime(x):
    return 2*np.sin(x)+2*np.sin(x*2)+5+np.sin(x/2)

xdbtime = np.arange(0,np.pi*4,0.1)
ydbtime = dbtime(xdbtime)

plt.grid()
plt.xlim(0,10)
plt.ylim(0,10)
plt.title("Fonction dbtime")

plt.plot(xdbtime,ydbtime)

plt.savefig('dbtime1.png')
