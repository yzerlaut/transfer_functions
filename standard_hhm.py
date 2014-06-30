from scipy import *
from pylab import *


g_pot = 36 # maximum potassium conductance
g_sod=120
g_lea=0.03
e_k = -77 # Potassium reversal potential
e_na=50
e_l=-54.387

cm=1

# Initial conditions
v_0 = -65 
n_0 = 0.3177
h_0 = 0.5961
m_0 = 0.0529

t_max = 100
dt = 0.01

def euler(f_func, x_0, t_max, dt):
  """Euler method for solving systems of differential equations"""
  x = np.asarray(x_0)
  res = [x_0]
  for t in np.arange(dt,t_max,dt):
      dx = f_func(x, t)
      x = x + np.asarray(dx)*dt
      res.append(x)
  return np.array(res)


def alpha_n(v):
   """opening rate"""
   res = where(v==-55,0.1,(0.01*(v+55)/(1-exp(-0.1*(v+55)))))
   return res
def alpha_m(v):
   """opening rate"""
   res = where(v==-40,1.0,(0.1*(v+40)/(1-exp(-0.1*(v+40)))))
   return res

def alpha_h(v):
   """opening rate"""
   res = 0.07*exp(-0.05*(v+65))
   return res

def beta_n(v):
   """closing rate"""
   return (0.125*exp(-0.0125*(v+65)))

def beta_m(v):
   """closing rate"""
   return (4*exp(-0.0556*(v+65)))

def beta_h(v):
   """closing rate"""
   return 1/(1+exp(-0.1*(v+35)))

def n_inf(v):
   """steady state activation"""
   return (alpha_n(v)/(alpha_n(v)+beta_n(v)))

def m_inf(v):
   """steady state activation"""
   return (alpha_m(v)/(alpha_m(v)+beta_m(v)))

def h_inf(v):
   """steady state activation"""
   return (alpha_h(v)/(alpha_h(v)+beta_h(v)))

def tau_n(v):
   """activation time constant"""
   return 1/(alpha_n(v)+beta_n(v))

def tau_m(v):
   """activation time constant"""
   return 1/(alpha_m(v)+beta_m(v))

def tau_h(v):
   """activation time constant"""
   return 1/(alpha_h(v)+beta_h(v))

def gate_n(v, n):
   """rate of activation change"""
   return alpha_n(v)*(1-n)-(beta_n(v)*n)

def gate_m(v, m):
   """rate of activation change"""
   return alpha_m(v)*(1-m)-(beta_m(v)*m)

def gate_h(v, h):
   """rate of activation change"""
   return alpha_h(v)*(1-h)-(beta_h(v)*h)

def pot_curr(v, n):
   """potassium current"""
   return g_pot*(n**4)*(e_k-v)

def sod_curr(v, m, h):
   """sodium current"""
   return g_sod*(m**3)*h*(e_na-v)

def lea_curr(v):
   """leak current"""
   return g_lea*(e_l-v)

def i_e(t):
   """external current"""
   return 10

# exercise 1b

def handm(x_vec, t):
   """system of differential equations describing sodium gating variables"""
   h, m = x_vec
   v = getV(t)
   dm = gate_m(v,m)
   dh = gate_h(v,h)
   return dh, dm

v_c = -20

def my_getV(t):
  if t >= 2:
      return v_c
  else:
      return -65

getV = np.vectorize(my_getV)

ts=arange(0,t_max,dt)

# 1a
vs=arange(-150,150,1)
figure(1)
clf()
subplot(2,1,1)
plot(vs,m_inf(vs),label='m')
plot(vs,h_inf(vs),label='h')
plot(vs,n_inf(vs),label='n')
xlabel('mV')
ylabel('Steady-state activation')
legend()
subplot(2,1,2)
plot(vs,tau_m(vs),label='m')
plot(vs,tau_h(vs),label='h')
plot(vs,tau_n(vs),label='n')
xlabel('mV')
ylabel('Time constant')
legend()

t_max = 20
dt = 0.01
ts=arange(0,t_max,dt)

m_0 = 0.0529
h_0 = 0.5961

# 1b
figure(2)
mr = euler(handm, [h_0, m_0], t_max, dt)
plot(ts, sod_curr(getV(ts), mr[:,1], mr[:,0]))
title('Sodium current')
xlabel('Time [ms]')

# 1c
figure(3)
clf()
for v_c in arange(-80,80,40):
   mr = euler(handm, [h_0, m_0], t_max, dt)
   plot(ts, sod_curr(getV(ts), mr[:,1], mr[:,0]),label=str(v_c))
title('Sodium current')
xlabel('Time [ms]')
legend()

show()

