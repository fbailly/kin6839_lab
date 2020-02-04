from casadi import *
import matplotlib.pyplot as plt
from IPython import embed
import numpy as np 
import biorbd            as brbd
from BiorbdViz           import BiorbdViz
import time

#plt.ion()

T = 1
N =  100               # number of control intervals
x = MX.sym('x',4)  # state trajectory
# pos  = X[0,:]
# vel   = X[1,:]
# th = X[2,:]
# w = X[3,:]
u = MX.sym('u')     # control trajectory (throttle)

m1 = 1
m2 = 1
g  = 9.81
l  = 1


f = lambda x,u: vertcat(x[1],(l*m2*sin(x[2])*x[3]**2+u+m2*g*cos(x[2])*sin(x[2]))/(m1+m2*(1-cos(x[2])**2)),x[3],-(l*m2*cos(x[2])*sin(x[2])*x[3]**2-u*cos(x[2])+(m1+m2)*g*sin(x[2]))/(l*m1+l*m2*(1-cos(x[2])**2)))
L = lambda u: u*u
F = Function('F', [x,u], [f(x,u)])
# Formulate discrete time dynamics
dae = {'x':x, 'p':u, 'ode':F(x,u) , 'quad':L(u)}
opts = {'tf':T/N,'number_of_finite_elements' : 20}
# opts = {'tf':T/N,"linear_solver" :"csparse"}
INT = integrator('INT', 'rk', dae, opts)

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
g=[]
lbg = []
ubg = []
# "Lift" initial conditions
Xk = MX.sym('X0', 4)
w += [Xk]
lbw += [0,0,0,0]
ubw += [0,0,0,0]
w0  += [0,0,0,0]
J   = 0

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = MX.sym('U_' + str(k))
    w   += [Uk]
    lbw += [-1000]
    ubw += [1000]
    w0  += [0]
    J += Uk*Uk
    # Integrate till the end of the interval
    Fk = INT(x0 = Xk, p = Uk)
    Xk_end = Fk['xf']

    # New NLP variable for state at end of interval
    Xk = MX.sym('X_' + str(k+1), 4)
    if k+1 == N :
        w   += [Xk]
        lbw +=  [-1,0,3,0]
        ubw +=  [-1,0,3,0]
        w0  +=  [-1,0,3,0]
    else :
        w   += [Xk]
        lbw += [-1000,-1000,-1000,-1000]
        ubw += [1000,1000,1000,1000]
        w0  += [0]*4
        # Add equality constraint
    g   += [Xk_end-Xk]
    lbg += [0]*4
    ubg += [0]*4

# Create NLP solver 
t = time.time()
opts = { 'ipopt.tol' : 1e-6, 'ipopt.constr_viol_tol': 1e-6,'ipopt.hessian_approximation':'limited-memory'}
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob, opts)
print(f"Time to create regular problem {time.time()-t}")

t = time.time()
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
print(f"Time to solve regular problem {time.time()-t}")
sol_opt = sol['x'].full().flatten()


pos_opt   = sol_opt[0::5]
vel_opt    = sol_opt[1::5]
th_opt  = sol_opt[2::5]
w_opt  = sol_opt[3::5]
u_opt    = sol_opt[4::5]

plt.plot(th_opt,label="theta")
plt.plot(w_opt,label="omega")
plt.plot(vel_opt,label="car vel")
plt.plot(pos_opt,label="car pos")
plt.plot(u_opt,label="control")
plt.legend()
plt.show(block=True)

# visualize optim
qs = np.array([pos_opt,th_opt])
np.save("visual",qs.T)
b = BiorbdViz(model_path="../data/inverse_pendulum.bioMod")
b.load_movement(qs.T)
b.exec()

# # simulate control
# m = brbd.Model("/home/fbailly/devel/models/inverse_pendulum.bioMod")
# qs       = np.zeros((N,2))
# qdots    = np.zeros((N,2))
# qddots   = np.zeros((N,2))
# q    = np.array([pos_opt[0],th_opt[0]])
# qdot = np.array([vel_opt[0],w_opt[0]])
# dt = T/N
# for t in range(N) :
#     tau           =  np.array([u_opt[t],0])
#     qddot         = m.ForwardDynamics(q, qdot, tau)
#     qdot         += qddot.to_array()*dt
#     q            += qdot*dt
#     qs[t,:]       = q
#     qdots[t,:]    = qdot
#     qddots[t,:]   = qddot.to_array()


# plt.plot(qs[:,1],label="theta")
# plt.plot(qdots[:,1],label="w")
# plt.plot(qdots[:,0],label="car vel")
# plt.plot(qs[:,0],label="car pos")
# plt.plot(qddots[:,0],label="control")
# plt.legend()
# plt.show(block=True)

# np.save("simu",qs)
# embed()

