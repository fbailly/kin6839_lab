#from sys import path
#path.append(r"<yourpath>/casadi-py27-v3.5.1")
from casadi import *
import matplotlib.pyplot as plt

T  = 10
N  = 100
dN = T/N
x  = MX.sym("x",2) #x[0] = p, x[1] = v
u  = MX.sym("u") # u[0] = a

def dyn_fun(x,u) :
    p = x[0]
    v = x[1]
    a = u
    return vertcat(v,a)

def int_fun(xk,uk) :
    M = 10
    state = xk
    control = uk
    for i in range(M):
        dstate = dyn_fun(state,control)
        state  += dstate * dN/M
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
	J += Uk*Uk
	w   += [Uk]
	lbw += [-2]
	ubw += [2]
	w0  += [0]

    # Integrate till the end of the interval
	Fk = int_fun(Xk, Uk)
	Xk_end = Fk

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
opts = {'ipopt.linear_solver' : 'mumps', 'ipopt.tol' : 1e-6, 'ipopt.constr_viol_tol': 1e-6, 'ipopt.hessian_approximation' : 'limited-memory'}
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob, opts)

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

p_opt  = w_opt[0::3]
v_opt  = w_opt[1::3]
u_opt  = w_opt[2::3]

plt.ion()
plt.plot(p_opt)
plt.plot(v_opt)
plt.plot(u_opt)
plt.show(block=True)