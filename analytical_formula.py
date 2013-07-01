#!/usr/bin/python
# Filename: analytical_formula.py

import numpy as np
from scipy import integrate

def tau_eff(fe, fi, params):
    """
    args : params
    returns : the efective membrane time constant
    """
    Cm = params['Cm'];Gl=params['Gl'];El=params['El']
    Qe = params['Qe'];Ne = params['Ne'];Te=params['Te'];Ee=params['Ee']
    Qi = params['Qi'];Ni = params['Ni'];Ti=params['Ti'];Ei=params['Ei']

    ge=fe*Ne*Qe*Te
    gi=fi*Ni*Qi*Ti
    return Cm/(Gl+ge+gi)

def campbell_theorem_single_exp(f, Q, T):
    """
    returns the mean and variance
    of a shotnoise convoluted by a single exponential
    args : f, Q, T // frequency, quantal, time decay
    returns : mean and variance
    """
    mu = f*Q*T
    sigma =  Q*np.sqrt(f*T/2)
    return mu, sigma

def rudolph_rho_v(V, param_rudolph):
    """
    args : v, param_rudolph
    param_rudolph = [Cm, Gl, El, mu_ge, s_ge, Te, Ee, mu_gi, s_gi, Ti, Ei]
    returns : the distribution for the membrane potential as the one derived
    in Rudolph et al. 2005 (with corrections)
    !!! IT NEEDS TO BE RENORMALIZED !!!
    """
    Cm = param_rudolph['Cm'];Gl=param_rudolph['Gl'];El=param_rudolph['El']
    ge = param_rudolph['mu_ge'];sge = param_rudolph['s_ge'];Te=param_rudolph['Te'];Ee=param_rudolph['Ee']
    gi = param_rudolph['mu_gi'];sgi = param_rudolph['s_gi'];Ti=param_rudolph['Ti'];Ei=param_rudolph['Ei']

    if ge.any()<0 or gi.any()<0:
        print "problem with the conductance traces, <0"
        break
    
    Tm = Cm/(ge+gi+Gl) # effective membrane time constant

    Te = 2*Te*Tm/(Te+Tm) # correction
    Ti = 2*Ti*Tm/(Ti+Tm) # see rudolph et al. 2005

    A1 = - (2*Cm*(ge+gi)+2*Cm*Gl+sge**2*Te+sgi**2*Ti)
    A1 /= 2*(sge**2*Te+sgi**2*Ti)
    A2 = 2*Cm*(Gl*(sge**2*Te*(El-Ee)+sgi**2*Ti*(El-Ei))+\
               (ge*sgi**2*Ti-gi*sge**2*Te)*(Ee-Ei))
    A2 /= (Ee-Ei)*np.sqrt(sge**2*Te*sgi**2*Ti)*(sge**2*Te+sgi**2*Ti)
    a_ln = sge**2*Te/Cm*(V-Ee)**2+sgi**2*Ti/Cm*(V-Ei)**2
    a_arctan = sge**2*Te*(V-Ee)+sgi**2*Ti*(V-Ei)
    a_arctan /= (Ee-Ei)*np.sqrt(sge**2*Te*sgi**2*Ti)
    return np.exp(A1*np.log(a_ln)+A2*np.arctan(a_arctan))

def rudolph_rho_normed(V,param_rudolph):
    integrand = lambda x: rudolph_rho_v(x, param_rudolph)
    N,error = integrate.quad(integrand,-np.inf,np.inf) # normalization factor
    print "absolute error on the function :",error
    return rudolph_rho_v(V, param_rudolph)/N

def rudolph_rho_normed_shotnoise(V, fe, fi, params):
    """
     here we return the same as in 'rudolph_rho_normed',
    we just adapt the formalism so that it takes shot-noise parameters
    """
    param_rudolph = params # here we copy the params dict() !!!

    param_rudolph['mu_ge'], param_rudolph['s_ge'] = \
      tf.campbell_theorem_single_exp(
          params['Ne']*fe, params['Qe'], params['Te'])
    param_rudolph['mu_gi'], param_rudolph['s_gi'] = \
      tf.campbell_theorem_single_exp(
          params['Ni']*fi, params['Qi'], params['Ti'])

    return rudolph_rho_normed(V, param_rudolph)

def proba_rudolph(param_rudolph):
    
    threshold = params['Vthre']
    # version with analytic integration
    integrand = lambda x: rudolph_rho_v(x, param_rudolph)
    N,error = integrate.quad(integrand, -np.inf, np.inf) # normalization factor
    
    return integrate.quad(integrand, threshold, np.inf)[0]/N

def proba_rudolph_shotnoise(fe, fi, params):
    """
     here we return the same as in 'rudolph_rho_normed',
    we just adapt the formalism so that it takes shot-noise parameters
    """
    param_rudolph = params # here we copy the params dict() !!!

    param_rudolph['mu_ge'], param_rudolph['s_ge'] = \
      tf.campbell_theorem_single_exp(
          params['Ne']*fe, params['Qe'], params['Te'])
    param_rudolph['mu_gi'], param_rudolph['s_gi'] = \
      tf.campbell_theorem_single_exp(
          params['Ni']*fi, params['Qi'], params['Ti'])

    return proba_rudolph(param_rudolph)

def kuhn_func_shotnoise(fe, fi, params):
    rate =  proba_rudolph_shotnoise(fe, fi, params)
    rate /= tau_eff(fe, fi, params)
    return rate
    
    
