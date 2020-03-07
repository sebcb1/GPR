
import numpy as np
import matplotlib.pyplot as plt

def dbtime(x):
	return (x/2-2)*(x/2-2)+2
	
xdbtime = np.arange(0,np.pi*4,0.1)
ydbtime = dbtime(xdbtime)

plt.grid()
plt.xlim(0,10)
plt.ylim(0,10)
plt.title("Fonction dbtime 2")

plt.plot(xdbtime,ydbtime)

plt.savefig('dbtime2.png')
