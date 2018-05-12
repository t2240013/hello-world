import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider, Button
import csv

#############################
###     Sub Functions     ###
#############################

def callback_time_cut(val):
    """
    time virtical axline. pow/mag vs freq (read_freq plot)
    """
    global plot_mode
    global val_time
    last_plot_mode = plot_mode
    plot_mode = 'time_cut'
    val_time = int(val)
    update_num_shadow(int(sneighbors.val))
    # plot 121
    lcuttime.set_xdata( [val, val] )
    lcuttime.set_alpha( alpha_ax )
    lcutfreq.set_alpha( 0.0 )
    # plot 122
    if plot_mode == last_plot_mode:
        replot_flags = get_replot_flag( val_time )
        replot_shadow( replot_flags )
        update_shadow( ~replot_flags )
        update_light()
    else:
        replot_shadow( [True, True ] )
        replot_light()
#        reform_axis( axis_freq[linlog], 'freq_cut')
    fig.canvas.draw_idle()

def callback_freq_cut(val):
    """
    freq holizontal axline. # pow/mag vs time
    """
    global plot_mode
    global val_freq
    last_plot_mode = plot_mode
    plot_mode = 'freq_cut'
    print( 'scale_freq', scale_freq)
    val_freq = sbval_to_idx( val, scale_freq )
    freqf = val_freq * scale_freq
    print( 'val val_freq freqf', val, val_freq, freqf )
    update_num_shadow(int(sneighbors.val))
    #plot 121
    lcutfreq.set_ydata( [freqf, freqf])
    lcuttime.set_alpha( 0.0 )
    lcutfreq.set_alpha( alpha_ax )
    #plot 122
    if plot_mode == last_plot_mode:
        replot_flags = get_replot_flag( val_freq )
        replot_shadow( replot_flags )
        update_shadow( ~replot_flags )
        update_light()
    else:
        replot_shadow( [True, True])
        replot_light()
#        reform_axis( axis_time[linlog], 'time_cut')
    fig.canvas.draw_idle()

def callback_neighbors(val):
    update_num_shadow(int(val))
    replot_shadow( [ True, True ] )
    fig.canvas.draw_idle()

def callback_alpha(val):
    global alpha_scale
    alpha_scale = val * val
    replot_shadow( [ True, True ] )
    fig.canvas.draw_idle()

def callback_linlog(event):
    global linlog
    linlog = (linlog+1)%2
    replot_light()
    replot_shadow( [True, True] )
    fig.canvas.draw_idle()

def update_light():
    if plot_mode == 'freq_cut':
        update_line( l2, xdata_freq, zdata.T[:,val_freq] )
        update_color( l2, color_freq_light, alpha_freq_light )
        update_marker_color( l2, color_freq_light )
#        reform_axis( axis_time[linlog], 'time_cut')
    else:
        update_line( l2, xdata_time, zdata[:,val_time] )
        update_color( l2, color_time_light, alpha_time_light )
        update_marker_color( l2, color_time_light )
#        reform_axis( axis_freq[linlog], 'freq_cut' )

def update_shadow( replot_flags ):
    for ilh in range(len(replot_flags) ):
        if replot_flags[ilh]:
            if plot_mode == 'freq_cut':
                update_shadow_ilh( ilh, val_freq, xdata_freq, zdata.T )
            else:
                update_shadow_ilh( ilh, val_time, xdata_time, zdata )

def update_shadow_ilh( ilh, val, xdata, ydata ):
    shadow_list = get_shadow_list( val, num_lower, num_higher )
    if shadow_list[ilh]:
        for ( lval, lo ) in zip( shadow_list[ilh], l2shadow[ilh] ):
#            print plot_mode, val, lo
            update_line( lo, xdata, ydata[:,lval] )

def replot_light( ):
    global l2
    if l2:
        l2.remove()
    if plot_mode == 'freq_cut':
        col = color_freq_light
        l2, = ax2.plot( xdata_freq, zdata[val_freq,:], '-o',
            	color=col, markerfacecolor=col, markeredgecolor=col, lw=1, ms=2 )
        reform_axis( axis_time[linlog], 'time_cut')
    else:
        col = color_time_light
        l2, = ax2.plot( xdata_time, zdata[:,val_time], '-o',
            	color=col, markerfacecolor=col, markeredgecolor=col, lw=1, ms=2 )
        reform_axis( axis_freq[linlog], 'freq_cut')

    ax2.set_yscale(yscale[linlog])
    l2.set_zorder(200)

def replot_shadow( replot_flags ):
    global l2shadow

    for ilh in range(len(replot_flags) ):
        if replot_flags[ilh]:
            # delete previous lines
            if l2shadow[ilh]:
                for lineobj in l2shadow[ilh]:
                    lineobj.remove()
                l2shadow[ilh] = []
            # Write new lines
            if plot_mode == 'freq_cut':
                replot_shadow_ilh( ilh, val_freq, xdata_freq, zdata.T,
                                   color_freq_shadow[ilh], alpha_freq_shadow[ilh] * alpha_scale )
            else:
                replot_shadow_ilh( ilh, val_time, xdata_time, zdata,
                                   color_time_shadow[ilh], alpha_time_shadow[ilh] * alpha_scale )

def replot_shadow_ilh( ilh, val, xdata, ydata, color, alpha ):
    shadow_list = get_shadow_list( val, num_lower, num_higher )
    if shadow_list[ilh]:
        l2shadow[ilh] = ax2.plot( xdata, ydata[:,shadow_list[ilh]], '-', color=color, lw=1, alpha=alpha )


def update_line( lo, xdata, ydata ):
    lo.set_xdata( xdata )
    lo.set_ydata( ydata )

def update_color( lo, color, alpha ):
    lo.set_color( color )
    lo.set_alpha( alpha )

def update_marker_color( lo, color ):
    lo.set_markerfacecolor( color )
    lo.set_markeredgecolor( color )

def reform_axis( axis, label ):
    ax2.axis( axis )
    ax2.set_xlabel( label )

def get_shadow_list( val, num_lower, num_higher ):
    lower_list =  range( max( 0, val - num_lower ), val )
    higher_list =  range( val+1, min( num_cut[plot_mode], val + 1 + num_higher ) )
    return  [ lower_list, higher_list ]

def get_replot_flag( val ):
    shadow_list = get_shadow_list( val, num_lower, num_higher )
    flags = []
    for ilh in range(len(l2shadow) ):
        if len(l2shadow[ilh]) == len(shadow_list[ilh]):
            flags.append( False )
        else:
            flags.append( True )
#    print flags
    return np.array( flags )

def update_num_shadow(num_shadow_all):
    global num_lower
    global num_higher
    if plot_mode == 'freq_cut':
        val = val_freq
        val_max = num_freq - 1
    else:
        val = val_time
        val_max = num_time - 1
    num_lower = min( val, int(num_shadow_all/2) )
    num_higher = num_shadow_all - num_lower
    hami = num_higher - ( val_max - val  )
    if( hami > 0 ):
        num_lower += hami
        num_higher -= hami

def sbval_to_idx( val, scale ):
    return int( round( val / scale ) )  # "%.1f" in Slidebar applies unexpected round ...

def onclick_motion(event):
    if ( drag == True ) and ( ax1 == event.inaxes ) :
        update_slider( event )

def onclick_press(event):
    global drag
    drag = True
    if ( ax1 == event.inaxes ):
        update_slider( event )

def onclick_release(event):
    global drag
    drag = False

def on_key(event):
    if event.key == 'right':
        stime.set_val( min( stime.val+1, stime.valmax ) )
    if event.key == 'left':
        stime.set_val( max( stime.val-1, stime.valmin ) )
    if event.key == 'up':
        sfreqf.set_val( min(sfreqf.val + scale_freq, sfreqf.valmax) )
    if event.key == 'down':
        sfreqf.set_val( max(sfreqf.val - scale_freq, sfreqf.valmin) )

def on_scroll(event):
    if event.step > 0:
        if plot_mode == 'time_cut':
            stime.set_val( min( stime.val+1, stime.valmax ) )
        else:
            sfreqf.set_val( min(sfreqf.val + scale_freq, sfreqf.valmax) )
    else:
        if plot_mode == 'time_cut':
            stime.set_val( max( stime.val-1, stime.valmin ) )
        else:
            sfreqf.set_val( max(sfreqf.val - scale_freq, sfreqf.valmin) )

def update_slider( event ):
    if plot_mode == 'time_cut':
        stime.set_val( min(event.xdata, stime.valmax) )
    else:
        sfreqf.set_val( min(event.ydata, sfreqf.valmax) )

def reset(event):
    global plot_mode
    stime.reset()
    sfreqf.reset()
    salpha.reset()
    sneighbors.reset()
    plot_mode = 'time_cut'
    update_shadow( [True, True ] )
    update_light()

def make_test_data():
    xx = np.linspace( 0, 7.5 * np.pi, 64 ).reshape(1,64)
    step = np.pi/2/192
    for i in range(1,192):
        xx = np.vstack( [ xx, xx[i-1,:] + step ] )
    yy =  np.cos(xx.T)**2 * 100
    f = open('wave2d.csv', 'w')
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(yy)
    f.close()
    return yy

def read_csv_file( filename ):
    """
    Read CSV file
    in : filename # line:time column:freq
    return : data # (time,freq)
    """
    zdata = np.genfromtxt( filename, dtype=float, delimiter="," ) 
    return zdata

def read_freq_file( filename ):
    fp = open( filename, 'r')
    data = fp.readlines()
    zdata= [ hoge.rstrip().replace(',',' ').split() for hoge in data ]
    if len(zdata[0]) > 200 :
        num_time = 256
        num_freq = 64
    else:
        num_time = 192
        num_freq = 64

    if zdata[0][0].isdigit():
        data_offset = 0
    else:
        data_offset = 1

    zdata = np.array( zdata[data_offset:data_offset+num_freq] )
    zdata = zdata[:,-num_time:].astype(np.int32)
    fp.close()
    return zdata

###########################
###        Main         ###
###########################
if __name__ == '__main__':
    # make test data
#    zdata = make_test_data()
    zdata = read_csv_file('wave2d.csv')
    print( 'zdata.shape=', zdata.shape )
#    zdata = read_freq_file('wave2d.csv')
    num_freq = zdata.shape[0]
    num_time = zdata.shape[1]
    num_cut = { 'freq_cut':num_freq,     # virtical (y)axis
               'time_cut':num_time}      # holizontal (x)axis
    zdata[50,:] /= 10.0
    zdata[20,:] /= 10.0
    zdata[:,40] /= 10.0
    zdata[:,160] /= 10.0

#    zlim = [ max(zdata.ravel())*1.1, 10**( np.floor(np.log10(max(zdata.ravel()))) + 1 ) ]
    zmax_log = 10**( np.floor(np.log10(np.max(zdata) ) ) + 1 )
    zmax_lin = np.max(zdata)*1.1

    # time variables
    val_time0 = 0
    val_time = val_time0
#    scale_freq = 0.1
    scale_freq = 0.2
#    color_time_light = 'blue'
#    color_time_light = 'dodgerblue'
    color_time_light = 'navy'
#    color_time_shadow = [ 'lightsteelblue', 'lightblue' ]
#    color_time_shadow = [ color_time_light, color_time_light ]
    color_time_shadow = [ 'dodgerblue', 'dodgerblue' ]
    alpha_time_light = 1.0
#    alpha_time_shadow = [ 0.3, 0.3]
    alpha_time_shadow = [ 1.0, 1.0]
    axis_freq = [ [0, num_freq * scale_freq, 1, zmax_lin ],
                [0, num_freq * scale_freq, 1, zmax_log ] ]
    xdata_time = np.arange(0.0, num_freq * scale_freq, scale_freq )

    # freq variables
    val_freq0 = 0
    val_freq = val_freq0
#    color_freq_light = 'green'
#    color_freq_light = 'mediumspringgreen'
    color_freq_light = 'green'
#    color_freq_shadow = [ 'sage', 'lightgreen' ]
#    color_freq_shadow = [ color_freq_light, color_freq_light ]
#    color_freq_shadow = [ 'mediumspringgreen', 'mediumspringgreen' ]
    color_freq_shadow = [ 'limegreen', 'limegreen' ]
    alpha_freq_light = 1.0
#    alpha_freq_shadow = [ 0.2, 0.4]
    alpha_freq_shadow = [ 1.0, 1.0]
    axis_time = [ [0, num_time, 1, zmax_lin ],
                 [0, num_time, 1, zmax_log ] ]
    xdata_freq = np.arange(0, num_time, 1)

    # position line
    alpha_ax = 1.0
    cmap_name='copper'
#    cmap_name='gray'
#    cmap_name='pink'

    # Parameter
    plot_mode = 'time_cut'
    num_higher = 0
    num_lower = 0
    alpha_scale = 1.0
    linlog = 1 # 0:Linear plot, 1:Semilog plot
    yscale = [ 'linear', 'log']

    # Setup plot space
#    fig = plt.figure(figsize=[14,7])
    fig = plt.figure(figsize=[12,7])
#    plt.subplots_adjust( bottom=0.20 )

    # Subplot 121
    ax1 = plt.subplot(121)
    X,Y = np.meshgrid( range(zdata.shape[1]+1), range(zdata.shape[0]+1) )
    Y = Y * scale_freq
#    im = plt.imshow(zdata, aspect='auto', origin='lowier', interpolation='none', cmap=cm.get_cmap(name=cmap_name) )
#    im = plt.pcolormesh(X, Y, zdata, snap=True, cmap=cm.get_cmap(name=cmap_name) )
    im = plt.pcolormesh(X, Y, zdata, snap=True, cmap=cmap_name )
    plt.ylabel('freq')
    plt.xlabel('time')
    lcuttime = plt.axvline( val_time, color=color_time_light, alpha=alpha_ax )
    lcutfreq = plt.axhline( val_freq, lw=1, color=color_freq_light, alpha=alpha_ax )
    plt.axis( [ X[0,:].min(), X[0,:].max(), Y[:,0].min(), Y[:,0].max() ] )

    # Subplot 122
    ax2 = plt.subplot(122)
    l2shadow = [[],[]]
    replot_shadow( [True, True])
    l2 = []
    replot_light( )
    plt.grid(which='major',color='gray',linestyle='-')
    plt.grid(which='minor',color='gray',linestyle='-')
    plt.axis( axis_freq[linlog])
    plt.ylabel('Pow/Mag')
    plt.xlabel('freq')

    # Slider
    axcolor = 'lightgoldenrodyellow'
    axfreq = plt.axes([0.1, 0.05, 0.3, 0.03], facecolor=axcolor)
    axtime = plt.axes([0.1, 0.10, 0.3, 0.03], facecolor=axcolor)
    axhigher = plt.axes([0.6, 0.05, 0.3, 0.03], facecolor=axcolor)
    axlower = plt.axes([0.6, 0.10, 0.3, 0.03], facecolor=axcolor)

    stime = Slider(axtime, 'time', 0, num_time-1, valinit=val_time, valfmt='%d')
    sfreqf = Slider(axfreq, 'freq', 0.0, (num_freq-1) * scale_freq, valinit=val_freq * scale_freq, valfmt='%.1f')
    sneighbors = Slider(axlower, '# neighbors', 0, num_time-1, valinit=num_lower, valfmt='%d')
    salpha = Slider(axhigher, 'alpha', 0, 1, valinit=np.sqrt(alpha_scale), valfmt='%.2f')

    stime.on_changed(callback_time_cut)
    sfreqf.on_changed(callback_freq_cut )
    salpha.on_changed(callback_alpha)
    sneighbors.on_changed(callback_neighbors)

    # Reset button
#    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
#    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
#    button.on_clicked(reset)

    # Linear/Log  button
    linlogax = plt.axes([0.48, 0.05, 0.05, 0.03])
    button = Button(linlogax, 'Lin/Log', color=axcolor, hovercolor='0.975')
    button.on_clicked(callback_linlog)

    # Mouse
    drag = False
    cid_press = fig.canvas.mpl_connect('button_press_event', onclick_press)
    cid_release = fig.canvas.mpl_connect('button_release_event', onclick_release)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', onclick_motion)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    cid_scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)

#    plt.tight_layout( w_pad = 2.0 )
    plt.subplots_adjust( bottom=0.22, wspace=0.2, left=0.04, right=0.98, top=0.98 )
    plt.show()
