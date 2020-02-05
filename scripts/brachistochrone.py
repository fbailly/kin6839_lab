from casadi import *
import time
from IPython import embed
import matplotlib.pyplot as plt
import numpy as np

tf         = MX.sym('tf')
state      = MX.sym('state',3)
control    = MX.sym('control')
N          = 100
dN         = tf/N
plt.ion()

def func_ode(state,control):
	x = state[0]
	y = state[1]
	v = state[2]
	return vertcat(v*sin(control),v*cos(control),-9.81*cos(control))

def func_int(state,control) :
	M = 10
	for i in range(M):
		dState = func_ode(state,control)
		state  += dState*dN/M
	return state

# Objective term
L =  tf*tf # quadratic

# Formulate discrete time dynamics

### REGULAR PROBLEM
# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = tf*tf
g=[]
lbg = []
ubg = []

# "Lift" initial conditions
w += [tf]
lbw += [0]
ubw += [inf]
w0 += [1]
Xk = MX.sym('X0', 3)
w += [Xk]
lbw += [0.01]*3
ubw += [0.01]*3
w0  += [0.01]*3

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
	Uk = MX.sym('U_' + str(k))
	w   += [Uk]
	if k == (N-1) :
		lbw += [0.01]
		ubw += [np.pi-0.01]
		w0  += [0.01]
	else :
		lbw += [0.01]
		ubw += [np.pi-0.01]
		w0  += [0.01]

    # Integrate till the end of the interval
	Fk = func_int(Xk, Uk)
	Xk_end = Fk
	
	# New NLP variable for state at end of interval
	Xk = MX.sym('X_' + str(k+1), 3)
	if k+1 == N :
		w   += [Xk]
		lbw +=  [8,-5,0]
		ubw +=  [8, -5,inf]
		w0  +=  [0]*2 + [0.01]
	else :
		w   += [Xk]
		lbw += [-100]*2 + [0]
		ubw += [100]*3
		w0  += [0]*2 + [0.01]
		# Add equality constraint
	g   += [Xk_end-Xk]
	lbg += [0]*3
	ubg += [0]*3

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

tf_opt = w_opt[0]
x_opt  = w_opt[1::4]
y_opt  = w_opt[2::4]
v_opt  = w_opt[3::4]
u_opt  = w_opt[4::4]

plt.plot(x_opt,y_opt)
plt.title('XY minimal time trajectory')
plt.figure()
plt.plot(u_opt,'x')
plt.title('Optimal slope')
plt.figure()
plt.plot(v_opt)
plt.title('Optimal speed')
plt.show(block=True)
