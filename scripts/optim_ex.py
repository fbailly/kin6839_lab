from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed

"""
Solve the Rosenbrock problem, formulated as the NLP:
minimize     x^2 + 100*z^2
subject to   z+(1-x)^2-y == 0
Joel Andersson, 2015
"""

# Declare variables
x = SX.sym("x")
y = SX.sym("y")

# Formulate the NLP
f = (1-x)**2 + (y-x**2)**2
g = x**2 + y**2 - 1
nlp = {'x':vertcat(x,y), 'f':f, 'g':g}

# Create an NLP solver
solver = nlpsol("solver", "ipopt", nlp)

# Solve the Rosenbrock problem
res = solver(x0  = [2.5,3.0],
             ubg = 0,
             lbg = 0)

# Print solution
print()
print("%50s " % "Optimal cost:", res["f"])
print("%50s " % "Primal solution:", res["x"])

# Plot solution
x = np.linspace(-2,3,1000)
y = np.linspace(-2,3,1000)
X,Y = np.meshgrid(x,y)
F = (1-X)**2 + (Y-X**2)**2
G = X**2 + Y**2 - 1 
plt.contour(X,Y,F,100)
plt.plot(res["x"][0],res["x"][1],'ro')
plt.contour(X,Y,G,np.array([0]),colors = ['red'])
plt.xlim([-2,3])
plt.ylim([-2,3])
plt.show(block=True)
embed()
