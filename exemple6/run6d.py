
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def dbtime(x1,x2):
	return (x1/2-2)*(x1/2-2)+2 + 2*np.sin(x2)+2*np.sin(x2*2)+5+np.sin(x2/2)+2*np.sin(x2)+2*np.sin(x2*2)+5+np.sin(x2/2)

x1 = np.linspace(0, 10, 100)[:,None]
x2 = np.linspace(0, 10, 100)[:,None]

yref = 10000
x1ref = -1
x2ref = -1


for a in x1:
	for b in x2:
		y = dbtime(a,b)
		if y < yref:
			yref = y
			x1ref = a
			x2ref = b
print ("x1: %d", x1ref)
print ("x2: %d", x2ref)

x = np.linspace(0, 10, 30)
y = np.linspace(0, 10, 30)

X, Y = np.meshgrid(x, y)
Z = dbtime(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 60, cmap='binary')
#ax.plot_wireframe(X, Y, Z, color='black')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(45, 45)
plt.savefig('dbtime.png')