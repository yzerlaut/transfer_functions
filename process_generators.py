#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

# parameters
Gl= 1e-8 ; Cm= 0.2*1e-9; El = -65*1e-3 
Ne = 200 ; Qe = 6*1e-9 ; Te = 5*1e-3; Ee = 0*1e-3 ; fe = 4
Ni = 50 ; Qi = 64*1e-9 ; Ti = 10*1e-3; Ei = -80*1e-3 ; fi= 10
Vthre=-50*1e-3; refrac = 5*1e-3 ; 


def white_gaussian(dt, mu, sigma):
    """return a realisation of a gaussian white noise process"""
    
    while True:
        yield np.random.randn() * sigma + mu

def apply(func):
    """apply function lazily"""

    def _wrapper(generator):
        return (func(x) for x in generator)

    return _wrapper

def current(generator):
    
    stored_currents = []
    
    def _wrapper(v, t):
        new_value = next(generator)
        stored_currents.append(new_value)
        return new_value
   
    return _wrapper, stored_currents

def conductance(generator, e_reversal):
    
    stored_conductances = []
    
    def _wrapper(v, t):
        new_g = next(generator)
        stored_conductances.append(new_g)
        return new_g * (e_reversal - v)
   
    return _wrapper, stored_conductances

def psp_train(spike_list, e_reversal):
    #idea to be tested
    
    def _wrapper(v, t):
        return sum((np.exp(-(t-t_spike))*(t>t_spike) for t_spike in spike_list))
   
    return _wrapper

@apply
def rectify(x):
    """ take an array and repaces its negative points and 
    replace it with 0 values """
   
    return x*(x>0)

def leaky_iaf(tmax, dt, i_exc, i_inh, spiking_mech=True, max_spikes=np.inf):
    """ functions that solve the membrane equations for 2 time varying 
    excitatory and inhibitory conductances
    N.B. reversal potentials, membrane prop. should be global """

    last_spike = -np.inf # time of the last spike, for the refractory period
    max_steps = int(tmax/dt)
    
    spikes = []
    potential_trace = []
    
    v, t, n_spikes = El, 0, 0

    for i in xrange(max_steps):
        t = i*dt
        if (t - last_spike)>refrac:
            iexc = i_exc(v,t)
            iinh = i_inh(v,t)
            v = v + dt/Cm*( Gl*(El-v) + iexc + iinh)
        
        potential_trace.append(v)

        if v > Vthre and spiking_mech:
            last_spike = t
            spikes.append(last_spike)
            n_spikes += 1
            v = El # reset!!!
            if n_spikes == max_spikes:
                break
    
    t = np.arange(len(potential_trace))*dt
    potential_trace = np.array(potential_trace)

    return t, potential_trace, spikes


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    mu_exc, mu_inh = Qe*Te*Ne*fe, Qi*Ti*Ni*fi
    sigma_exc, sigma_inh = Qe*np.sqrt(Te*fe*Ne), Qi*np.sqrt(Ti*fi*Ni)
    E_exc, E_inh = 0, -80e-3
    dt = 0.1e-3

    gexc, exc_list = conductance(
                           rectify(
                               white_gaussian(dt, mu_exc, sigma_exc)
                           ), 0)
    
    ginh, inh_list = conductance(
                           rectify(
                               white_gaussian(dt, mu_inh, sigma_inh)
                           ), -80)
    
    t, v_m, spikes = leaky_iaf(2, dt, gexc, ginh)

    plt.plot(t, v_m)
    plt.show()
