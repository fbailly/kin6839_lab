#from sys import path
#path.append(r"<yourpath>/casadi-py27-v3.5.1")
from casadi import *
import matplotlib.pyplot as plt

x = MX.sym("x")
y = MX.sym("y")
f = (1-x)**2 + (y-x**2)**2
g = (x+1)**2 + (y+1)**2 - 1
#g =  0
nlp = {'x':vertcat(x,y), 'f':f, 'g':g}
solver = nlpsol("solver", "ipopt", nlp)
res = solver(x0  = [2.5,3.0],
             ubg = 0,
             lbg = 0)
# Print solution
print()
print("%50s " % "Optimal cost:", res["f"])
print("%50s " % "Primal solution:", res["x"])
x = np.linspace(-2,3,1000)
y = np.linspace(-2,3,1000)
X,Y = np.meshgrid(x,y)
F = (1-X)**2 + (Y-X**2)**2
G = (X+1)**2 + (Y+1)**2 - 1
plt.contour(X,Y,F,500)
plt.plot(res["x"][0],res["x"][1],'ro')
plt.contour(X,Y,G,np.array([0]),colors = ['red'])
plt.xlim([-2,3])
plt.ylim([-2,3])
plt.axis("equal")
plt.show(block=True)