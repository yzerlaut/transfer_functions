#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

# parameters
Gl= 1e-8 ; Cm= 0.2*1e-9; El = -65*1e-3 
Ne = 200 ; Qe = 6*1e-9 ; Te = 5*1e-3; Ee = 0*1e-3 ; fe = 4
Ni = 50 ; Qi = 64*1e-9 ; Ti = 10*1e-3; Ei = -80*1e-3 ; fi= 10
Vthre=-50*1e-3; refrac = 5*1e-3 ; 


def white_gaussian(tmax, dt, mu, sigma):
    """return a realisation of a gaussian white noise process"""
    
    for i in xrange(int(tmax/dt)):
        yield np.random.randn() * sigma + mu

def apply(func):
    """apply function lazily"""

    def _wrapper(generator):
        return (func(x) for x in generator)

    return _wrapper

@apply
def rectify(x):
    """ take an array and repaces its negative points and 
    replace it with 0 values """
   
    return out*(out>0)

def launch_sim_current(dt, current_generator, spiking_mech=True, max_spikes=np.inf):
    """ functions that solve the membrane equations for 2 time varying 
    excitatory and inhibitory conductances
    N.B. reversal potentials, membrane prop. should be global """

    last_spike = -np.inf # time of the last spike, for the refractory period
    
    spikes = []
    current_trace = []
    potential_trace = []
    
    v, t, n_spikes = El, 0, 0

    for i, I in enumerate(current_generator):
        t = i*dt
        if (t - last_spike)>refrac:
            v = v + dt/Cm*( Gl*(El-v) + I)
        
        potential_trace.append(v)
        current_trace.append(I)

        if v > Vthre and spiking_mech:
            last_spike = t
            spikes.append(last_spike)
            n_spikes += 1
            v = El # reset!!!
            if n_spikes == max_spikes:
                break
    
    t = np.arange(len(potential_trace))*dt
    potential_trace = np.array(potential_trace)
    current_trace = np.array(current_trace)

    return t, current_trace, potential_trace, spikes
