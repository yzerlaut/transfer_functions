import matplotlib.pylab as plt
import numpy as np
import cPickle



def plot_func(results):

    parameters = results['parameters']
    dt = parameters['dt']
    window = parameters['window']
    
    sta_filtered_ge = results['ge_filter']
    sta_filtered_gi = results['gi_filter']
    
    ge_bins, gi_bins, f_val = results['nonlinearity']
    rotation_vec = results['rotation_vector']

    rot_ge_bins, rot_gi_bins, rot_f_val = results['rotated_nonlinearity']

    
    time_lag = np.arange(len(sta_filtered_ge))*dt+window[0]

    f1 = plt.figure(figsize=(14,4))
    ax11 = plt.subplot(121)
    ax12 = plt.subplot(122)
    
    def plt_sta(subplot,sta_vec,color='b'):
        subplot.plot(time_lag*1e3, sta_vec,color=color)
        plt.xlabel('ms')

    plt_sta(ax11,sta_filtered_ge,color='b')
    plt_sta(ax12,sta_filtered_gi,color='r')



if __name__=='__main__':
    import sys, os
    _, filename = sys.argv
    with file(filename,'r') as fid:
        results = cPickle.load(fid)
    plot_func(results)
    plt.show()
    
    

