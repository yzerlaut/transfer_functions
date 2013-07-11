#!/usr/bin/env python
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt

def ornstein_uhlenbeck(dt, mu, sigma, tau):
    """return a realisation of ornstein-uhlenbeck process"""

    diffcoef = 2*sigma**2/tau
    A = np.sqrt(diffcoef*tau/2.*(1-np.exp(-2*dt/tau)))
    y0 = mu

    y = y0
    while True:
        noise = np.random.randn()
        y = y0 + (y - y0) * np.exp(-dt/tau) + A * noise
        yield y

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

def null(v, t):
    return 0

def conductance(generator, e_reversal):
    
    stored_conductances = []
    
    def _wrapper(v, t):
        new_g = next(generator)
        stored_conductances.append(new_g)
        return new_g * (e_reversal - v)
   
    return _wrapper, stored_conductances

def poisson_spike_generator(frequency):
    next_spike = 0
    while True:
        next_spike += np.random.exponential(1./frequency)
        yield next_spike

def conductance_shotnoise(spike_generator, kernel, e_reversal, **kernel_params):
    #idea to be tested
    stored_conductances = []
    spike_list = [next(spike_generator)]
    
    def exp(t,spike_list):
        spk_array = np.array(spike_list)
        return kernel_params['Q']*np.exp(-(t-spk_array)/kernel_params['T'])

    def alpha(t,spike_list):
        spk_array = np.array(spike_list)
        return kernel_params['Q']*(t-spk_array)/kernel_params['T']*\
                           np.exp(-(t-spk_array)/kernel_params['T'])
    
    kernel_functions = { 'exp' : exp ,'alpha' : alpha}

    def _wrapper(v, t):
        if t>=spike_list[-1]:
            nxt_spk = next(spike_generator)
            spike_list.append(nxt_spk)
            while nxt_spk<=t:
                nxt_spk = next(spike_generator)
                spike_list.append(nxt_spk)
        new_g = kernel_functions[kernel](t,spike_list[:-1]).sum()
        stored_conductances.append(new_g)
        return new_g * (e_reversal - v)
   
    return _wrapper, stored_conductances, spike_list

@apply
def rectify(x):
    """ take an array and repaces its negative points and 
    replace it with 0 values """
   
    return x*(x>0)

def leaky_iaf(tmax, dt, i_exc, i_inh=null, Cm=2e-10, Gl=1e-8, El=-65e-3,
              Vthre=-50e-3, Vreset=-60e-3, t_refrac = 5e-3, max_spikes=np.inf):
    """ functions that solve the membrane equations for 2 time varying 
    excitatory and inhibitory conductances
    args : tmax, dt, i_exc, i_inh, Cm, Gl, El,
    Vreset, Vthre, t_refrac, spiking_mech=True, max_spikes=inf
    returns : t, v, spikes
    """

    last_spike = -np.inf # time of the last spike, for the refractory period
    max_steps = int(tmax/dt)
    
    spikes = []
    potential_trace = []
    
    v, t, n_spikes = Vreset, 0, 0

    for i in xrange(max_steps):
        t = i*dt
        
        iexc = i_exc(v,t)
        iinh = i_inh(v,t)
        
        if (t - last_spike)>t_refrac:
            v = v + dt/Cm*( Gl*(El-v) + iexc + iinh)
        
        potential_trace.append(v)

        if v > Vthre:
            last_spike = t
            spikes.append(last_spike)
            n_spikes += 1
            v = Vreset
            if n_spikes == max_spikes:
                break
    
    t = np.arange(len(potential_trace))*dt
    potential_trace = np.array(potential_trace)

    return t, potential_trace, spikes


# if __name__ == '__main__':

#     import matplotlib.pyplot as plt

#     mu_exc, mu_inh = Qe*Te*Ne*fe, Qi*Ti*Ni*fi
#     sigma_exc, sigma_inh = Qe*np.sqrt(Te*fe*Ne), Qi*np.sqrt(Ti*fi*Ni)
#     E_exc, E_inh = 0, -80e-3
#     dt = 0.1e-3

#     gexc, exc_list = conductance(
#                            rectify(
#                                white_gaussian(dt, mu_exc, sigma_exc)
#                            ), 0)
    
#     ginh, inh_list = conductance(
#                            rectify(
#                                white_gaussian(dt, mu_inh, sigma_inh)
#                            ), -80)
    
#     t, v_m, spikes = leaky_iaf(2, dt, gexc, ginh)

#     plt.plot(t, v_m)
#     plt.show()
