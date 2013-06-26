import matplotlib.pylab as plt
import numpy as np
import cPickle
from mpl_toolkits.mplot3d import axes3d
from matplotlib import gridspec
from matplotlib import colors


def plot_func(results):

    # simulation parameters
    parameters = results['parameters']
    dt = parameters['dt']
    window = parameters['window']

    # stats ot the results
    stats = results['stats']
    firing_rate = stats['firing_rate']
    tstop = stats['tstop']
    spikes_number = stats['spike_number']
    print ""
    print "the firing frequency was : ",firing_rate
    print " for a number of spikes : ", spikes_number
    print ""

    sta_filtered_ge = results['ge_filter']
    sta_filtered_gi = results['gi_filter']
    
    ge_bins, gi_bins, f_val = results['nonlinearity']
    
    rotation_vec = results['rotation_vector']
    rot_matrix = np.array([rotation_vec, [-rotation_vec[1], rotation_vec[0]]])

    rot_ge_bins, rot_gi_bins, rot_f_val = results['rotated_nonlinearity']

    ### STA filters
    time_lag = np.arange(len(sta_filtered_ge))*dt+window[0]

    f1 = plt.figure(figsize=(14,4))
    ax11 = plt.subplot(121)
    ax12 = plt.subplot(122)
    
    def plt_sta(subplot,sta_vec,color='b'):
        subplot.plot(time_lag*1e3, sta_vec,color=color)
        plt.xlabel('ms')

    plt_sta(ax11,sta_filtered_ge,color='b')
    plt_sta(ax12,sta_filtered_gi,color='r')

    ### Non Linearity with the principal component lines
    f_val = f_val/dt # we put it in terms of frequency
    f2 = plt.figure(figsize=(8,5))
    ax21 = f2.add_subplot(111)
    plt.title('Non-Linearity with principal component axis')
    z_max = min(10,f_val.max()) # max 10 Hz !!! or below if not reached...
    img21  = ax21.contourf(ge_bins, gi_bins, f_val,np.linspace(0, z_max, 15))
    cb21 = plt.colorbar(img21)
    cb21.set_label('Hz')
    plt.xlabel('proj of ge')
    plt.ylabel('proj of gi')
    
    # now we add the principal companent lines
    # it's a bit tricky to insure that the lines are really on the plot !

    De = (ge_bins.max()-ge_bins.min())/10 # 10 lines
    for xshift in np.linspace(ge_bins.min(),ge_bins.max(),10):
        plt.plot([xshift,xshift+De],\
                 [gi_bins.min(), gi_bins.max()], 'w',lw=1)
    plt.ylim(gi_bins.min(),gi_bins.max())
    plt.xlim(ge_bins.min(),ge_bins.max())

    ## Non linearity after rotation along the principal component axis
    rot_f_val = rot_f_val/dt
    
    f3 = plt.figure()
    xsec, ysec = rot_ge_bins.mean(),rot_gi_bins.mean()

    ax_cont=plt.subplot(111)
    plt.contourf(rot_ge_bins, rot_gi_bins, rot_f_val, np.linspace(0,z_max,15))
    plt.axvline(xsec, color='w')
    plt.axhline(ysec, color='w')

    ## plotting the non linearity along those 
    f4 = plt.figure(figsize=(15,4))
    mymap = colors.LinearSegmentedColormap.from_list(\
                'mycolors',['blue','red'])
    print rotation_vec
    ge_steps = np.arange(1,len(ge_bins)-1,int(len(ge_bins)/10))
    plt.subplot(121)
    plt.title('along the first dimension')
    plt.ylabel('(Hz)')
    plt.xlabel('rotated projection')
    for ii in ge_steps:
        r = 1.*ii/len(ge_steps)
        current  = np.where(f_val[:,ii]>z_max,0,f_val[:,ii])
        plt.plot(rot_gi_bins,current,color = mymap(r,1),lw=2,alpha=.5)
    plt.xlim(rot_gi_bins.min(),rot_gi_bins.max())

    gi_steps = np.arange(1,len(gi_bins)-1,int(len(gi_bins)/10))
    plt.subplot(122)
    plt.title('along the second dimension')
    plt.ylabel('(Hz)')
    plt.xlabel('rotated projection')
    for ii in gi_steps:
        r = 1.*ii/len(gi_steps)
        current  = np.where(f_val[ii,:]>z_max,0,f_val[ii,:])
        plt.plot(rot_ge_bins,current,color = mymap(r,1),lw=2,alpha=.5)
    plt.xlim(rot_ge_bins.min(),rot_ge_bins.max())

    

if __name__=='__main__':
    import sys, os
    _, filename = sys.argv
    with file(filename,'r') as fid:
        results = cPickle.load(fid)
    plot_func(results)
    plt.show()
    
    

