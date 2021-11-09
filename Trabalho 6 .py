import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, solve_ivp
from scipy.fft import fft

def ode(t,y,q,omega_0,b):
    f = np.empty_like(y)
    f[0] = y[1]
    f[1] =(-q*y[1]) - np.sin(y[0]) + (b*np.cos(omega_0*t))
    return f

theta = np.radians(90)
omega = 0
q = 0.5
omega_0 = 2/3
v0 = omega_0 / (2*np.pi)
g=9.8
N = 100000
b = [0.9,1.15]
dt = np.linspace (0, 1000, N)
te = np.linspace (0, 1000, N)
y0 = [theta, omega]
ts = (te.min(), te.max())
s = []
plt.xlim(0,200) #ate 200 segundos
plt.ylim(-50,50)
s.append(solve_ivp (ode, t_span=ts, y0=y0, t_eval=te, args=(q,omega_0,b[0]), rtol=1.e-12, atol=1.e-12))
plt.plot(te,s[0].y[0],'green')
s.append(solve_ivp (ode, t_span=ts, y0=y0, t_eval=te, args=(q,omega_0,b[1]), rtol=1.e-12, atol=1.e-12))
plt.plot(te,s[1].y[0],'blue')
plt.show()

#Tentamos fazer a b mas não conseguimos.

nu_0 = omega_0/(2*np.pi)

cn=fft(s[0].y[0])
cn2=fft(s[1].y[0])

an = +2 * cn.real[:N // 2] / N
bn = -2 * cn.imag[:N // 2] / N