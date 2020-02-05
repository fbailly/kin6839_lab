from casadi import *
import time
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np

T = 2
state      = MX.sym('state',2)
control    = MX.sym('control')
N          = 100
dN         = T/N
plt.ion()

def func_ode(state,control):
	x = state[0]
	v = state[1]
	u = control
	return vertcat(v,u)

def func_int(state,control) :
	M = 10
	for i in range(M):
		dState = func_ode(state,control)
		state  += dState*dN/M
	return state

# Formulate discrete time dynamics

### REGULAR PROBLEM
# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

# "Lift" initial conditions
Xk = MX.sym('X0', 2)
w += [Xk]
lbw += [0.0]*2
ubw += [0.0]*2
w0  += [0.0]*2

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
	Uk = MX.sym('U_' + str(k))
	w   += [Uk]
	lbw += [-2]
	ubw += [2]
	w0  += [0]

    # Integrate till the end of the interval
	Fk = func_int(Xk, Uk)
	Xk_end = Fk
	J += Uk*Uk
	# New NLP variable for state at end of interval
	Xk = MX.sym('X_' + str(k+1), 2)
	if k+1 == N :
		w   += [Xk]
		lbw +=  [1,0]
		ubw +=  [1,0]
		w0  +=  [0]*2
	else :
		w   += [Xk]
		lbw += [-100]*2 
		ubw += [100]*2
		w0  += [0]*2 
		# Add equality constraint
	g   += [Xk_end-Xk]
	lbg += [0]*2
	ubg += [0]*2

# Create NLP solver 
t = time.time()
opts = {'ipopt.linear_solver' : 'mumps', 'ipopt.tol' : 1e-6, 'ipopt.constr_viol_tol': 1e-6, 'ipopt.hessian_approximation' : 'limited-memory'}
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob, opts)
print(f"Time to create regular problem {time.time()-t}")

t = time.time()
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
print(f"Time to solve regular problem {time.time()-t}")
w_opt = sol['x'].full().flatten()

p_opt  = w_opt[0::3]
v_opt  = w_opt[1::3]
u_opt  = w_opt[2::3]

plt.ion()
plt.plot(p_opt)
plt.plot(v_opt)
plt.plot(u_opt)
embed()
