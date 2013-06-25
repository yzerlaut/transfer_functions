import numpy as np
import tf_filters as tf
import cPickle


# cell parameters
Gl= 1e-8 ; Cm= 0.2*1e-9; El = -65*1e-3
Vthre=-50*1e-3; refrac = 5*1e-3 ; 
# stimulation parameters
Ne = 200 ; Qe = 6*1e-9 ; Te = 5*1e-3; Ee = 0*1e-3
Ni = 50 ; Qi = 64*1e-9 ; Ti = 10*1e-3; Ei = -80*1e-3

# SIMULATION and ANALYSIS parameters
dt = 1e-4
tstop = 500
max_spikes = 1500
nbins = 20
window = (-30*1e-3,0) # window for the sta analysis


### PARAMETER SCAN
# we scan over fe and fi, so variance and mean are actually related...

def pack_parameters(fe,fi):
    parameters ={
        'dt' : dt , 'window' : window, 
        'fe' : fe, 'Ne' : Ne, 'Qe' : Qe, 'Te' : Te,
        'fi' : fi, 'Ni' : Ni, 'Qi' : Qi, 'Ti' : Ti
        }
    return parameters
    
def Single_Trial(fe,fi):

    results = {} # dictionary for results
    results['parameters'] = pack_parameters(fe,fi)
    # conductance values
    ge_mu, ge_sigma = fe*Ne*Qe*Te, Qe*np.sqrt(fe*Ne*Te/2)
    gi_mu, gi_sigma = fi*Ni*Qi*Ti, Qi*np.sqrt(fi*Ni*Ti/2)
    # prepare the conductance vectors
    ge_trace = tf.ornstein_uhlenbeck(tstop, dt, ge_mu, ge_sigma, Te)
    gi_trace = tf.ornstein_uhlenbeck(tstop, dt, gi_mu, gi_sigma, Ti)
    ge_trace,gi_trace = tf.rectify(ge_trace), tf.rectify(gi_trace) ### non zero conductances !!

    # run the simulation
    t, v, spikes = tf.launch_sim(dt, ge_trace, gi_trace, max_spikes = 300)
    results['stats'] = {
        'firing_rate':len(spikes)/t.max(),
        'tstop':t.max(),
        'spike_number':len(spikes)
        }
    
    # calculate the filters
    lag_ge,sta_ge = tf.calculate_sta(dt, ge_trace, spikes, window)
    cov_ge = tf.estimate_cov_matrix(ge_trace, len(sta_ge))
    sta_filtered_ge = tf.whiten_filter(cov_ge,sta_ge)

    lag_gi,sta_gi = tf.calculate_sta(dt, gi_trace, spikes, window)
    cov_gi = tf.estimate_cov_matrix(gi_trace, len(sta_gi))
    sta_filtered_gi = tf.whiten_filter(cov_gi,sta_gi)

    results['ge_filter'] = sta_filtered_ge
    results['gi_filter'] = sta_filtered_gi
    
    # estimate two-dimensional nonlinearity in g_e/g_i space from traces and
    # filters
    spk_idx = (np.array(spikes)/dt).astype(int)
    ge_proj, ge_spike = tf._calc_projections(ge_trace, sta_ge, spk_idx)
    gi_proj, gi_spike = tf._calc_projections(gi_trace, sta_gi, spk_idx)

    ge_bins, gi_bins, f_val = tf.calculate_nonlinearity(
                ge_proj, gi_proj, ge_spike, gi_spike, nbins)

    results['nonlinearity'] = (ge_bins, gi_bins, f_val)

    ## Rotation of the non-linearity
    rotation_vec = tf.find_slope(ge_spike, gi_spike)
    rot_matrix = np.array([rotation_vec, [-rotation_vec[0], rotation_vec[1]]])

    results['rotation_vector'] = rotation_vec

    def rotate(x, y):
        return np.dot(rot_matrix.T, np.array([x,y]))
    
    rot_ge_proj, rot_gi_proj = rotate(ge_proj, gi_proj)
    rot_ge_spike, rot_gi_spike = rotate(ge_spike, gi_spike)

    rot_ge_bins, rot_gi_bins, rot_f_val = tf.calculate_nonlinearity(
        rot_ge_proj, rot_gi_proj, rot_ge_spike, rot_gi_spike, nbins)
    
    results['rotated_nonlinearity'] = (rot_ge_bins, rot_gi_bins, rot_f_val)

    return results


if __name__=='__main__':
    import sys, os
    _, fe, fi, path = sys.argv
    filename = "params_scan_fe%s_fi%s.pickle" %(fe, fi)
    path = os.path.join(path,filename)
    fe = float(fe)
    fi = float(fi)
    results = Single_Trial(fe,fi)
    with file(filename,'w') as fid:
        cPickle.dump(results,fid)
    

    
    
