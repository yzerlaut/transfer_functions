## PHYSICAL UNITS ###

# parameters
Gl= 1e-8 ; Cm= 0.2*1e-9; El = -65*1e-3 
Ne = 200 ; Qe = 6*1e-9 ; Te = 5*1e-3; Ee = 0*1e-3 ; fe = 4
Ni = 50 ; Qi = 64*1e-9 ; Ti = 10*1e-3; Ei = -80*1e-3 ; fi= 10
Vthre=-50*1e-3; refrac = 5*1e-3 ; 

import numpy as np
import matplotlib.pylab as plt
from scipy import linalg

def ornstein_uhlenbeck(tmax, dt, mu, sigma, tau):
    """return a realisation of ornstein-uhlenbeck process"""

    diffcoef = 2*sigma**2/tau
    y0 = mu
    n_steps = int(tmax/dt)
    A = np.sqrt(diffcoef*tau/2.*(1-np.exp(-2*dt/tau)))
    noise = np.random.randn(n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    for i in range(n_steps-1):
        y[i+1] = y0 + (y[i]-y0)*np.exp(-dt/tau)+A*noise[i]
    return y

def white_gaussian(tmax, dt, mu, sigma):
    """return a realisation of a gaussian white noise process"""
    
    return np.random.randn(int(tmax/dt)) * sigma + mu

def rectify(x):
    """ take an array and repaces its negative points and replace it with 0 values """
    out = x.copy()
    out[out<0]=0
    return out


def launch_sim(dt, ge, gi, spiking_mech=True):
    """ functions that solve the membrane equations for 2 time varying 
    excitatory and inhibitory conductances
    N.B. reversal potentials, membrane prop. should be global """
    
    try: 
        tstop = len(ge)*dt
    except TypeError:
        pass
    try: 
        tstop = len(gi)*dt
    except TypeError:
        pass
    
    t = np.arange(0,tstop,dt)
    v = np.ones(t.size)*El # refractory if not changed

    last_spike = -tstop # time of the last spike, for the refractory period
    spikes = []
    for i in range(t.size-1):
        if (t[i+1]-last_spike)>refrac:
            v[i+1] = v[i] + dt/Cm*( Gl*(El-v[i]) + gi[i]*(Ei-v[i]) + ge[i]*(Ee-v[i]) )
        if v[i+1]>Vthre and spiking_mech:
            last_spike = t[i+1]
            spikes.append(last_spike)
            #v[i+1]=0 ## UNCOMMENT to have a nice spike shape...
    return t, v, spikes

def calculate_sta(dt, x, spikes, window):
    """ function that calculate the STA of x 
    arguments : dt (float), x, spikes (np.array), window (tuple of minimum and maximum lag in time units)"""
    
    spike_train = np.histogram(spikes, np.arange(len(x))*dt)[0]
    min_index = int(window[0]/dt)
    max_index = len(spike_train) - int(window[1]/dt)
    sta = np.correlate(x, spike_train[-min_index-1:max_index]) ### TO BE CHECK
    lag = np.arange(len(sta))*dt+window[0]
    return lag,sta/len(spikes)
    
    
def ou_cov_matrix(sigma, tau, n_max, dt):
    """calculate and return covariance matrix of ornstein-uhlenbeck proccess"""
    time_lag = np.arange(n_max)*dt
    column = sigma**2*np.exp(-time_lag/tau)
    cov = linalg.toeplitz(column)
    return cov

def rotate_matrix(vec, mat):
    return np.dot(np.dot(vec, mat), vec.T)

def estimate_cov_matrix(y, n_max):
    """ argument : vector (np.array), int (number of interesting points in the past)
    returns the measured covariance matrix of a vector """
    
    y_nomean = y - np.mean(y)
    n_pts = len(y_nomean)
    cov = [np.mean(y_nomean[i:]*y_nomean[:n_pts-i])
          for i in range(n_max)]
    return linalg.toeplitz(cov)

def whiten_filter(cov_matrix, sta_filter):
    """ remove the signal correlations from the STA filters
    arguments : covariance matrix of the signal, STA filters """
        
    evals, evecs = np.linalg.eig(cov_matrix)
    d = 1./np.sqrt(evals)
    whitening_filter = np.dot(evecs,np.dot(np.diag(d),evecs.T))
    sta_whitened = np.dot(whitening_filter, sta_filter-sta_filter.mean())
    return sta_whitened
