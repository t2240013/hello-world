import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, Button, TextBox
import csv

#############################
###  Callback Functions   ###
#############################

def callback_time_cut(val):
    """
    time virtical axline. pow/mag vs freq (read_freq plot)
    """
    global plot_mode
    global idx_time
    last_plot_mode = plot_mode
    plot_mode = 'time_cut'
    idx_time = int(val)
    update_num_shadow(int(sneighbors.val))
    # plot 121
    lcuttime.set_xdata( [val, val] )
    lcuttime.set_alpha( alpha_hm )
    lcutfreq.set_alpha( 0.0 )
    # plot 122
    if plot_mode == last_plot_mode:
        replot_flags = get_replot_flag( idx_time ) # [True/False, True/False]
        replot_shadow( replot_flags )
        update_shadow( ~replot_flags )
        update_light()
    else:
        replot_shadow( [True, True ] )
        replot_light()
        reform_axis()
#        update_ticks_format( ax['plot'].xaxis, scale_freq )
#        reform_ticks()

    fig.canvas.draw_idle()

def callback_freq_cut(val):
    """
    freq holizontal axline. # pow/mag vs time
    """
    global plot_mode
    global idx_freq
    last_plot_mode = plot_mode
    plot_mode = 'freq_cut'
    print( 'scale_freq', scale_freq)
    idx_freq = freq_to_idx( val, scale_freq )
    val_freq = idx_freq * scale_freq
    print( 'val idx_freq val_freq', val, idx_freq, val_freq )
    update_num_shadow(int(sneighbors.val))
    #plot 121
    lcutfreq.set_ydata( [val_freq, val_freq])
    lcuttime.set_alpha( 0.0 )
    lcutfreq.set_alpha( alpha_hm )
    #plot 122
    if plot_mode == last_plot_mode:
        replot_flags = get_replot_flag( idx_freq )
        replot_shadow( replot_flags )
        update_shadow( ~replot_flags )
        update_light()
    else:
        replot_shadow( [True, True])
        replot_light()
        reform_axis()
#        update_ticks_format( ax['plot'].xaxis, 1.0 )
#        reform_ticks()
    
    fig.canvas.draw_idle()

def callback_neighbors(val):
    update_num_shadow(int(val))
    replot_shadow( [ True, True ] )
    fig.canvas.draw_idle()

def callback_alpha(val):
    global alpha_neighbors
    alpha_neighbors = val * val
    replot_shadow( [ True, True ] )
    fig.canvas.draw_idle()

def callback_linlog(event):
    global linlog
#    global ticks_freq
#    global ticklabels_freq    
    if linlog == 'linear':
        linlog = 'symlog'
    else:
        linlog = 'linear'
#    ticks_freq, ticklabels_freq = update_ticks_freq()    

#    replot_light()
#    replot_shadow( [True, True] )
    reform_axis()
#    update_ticks_format( ax['plot'].xaxis, get_current_xscale() )
#xx   reform_ticks()
    replot_heat()

    fig.canvas.draw_idle()

def callback_sf(text):
    global sf
    global scale_freq   
    global ax
    global sval_freq
    global xdata_freq
    global view_freq

    sf = eval(text)
    print( 'sf ', sf)
    scale_freq = get_scale_freq()
#    ticks_freq, ticklabels_freq = update_ticks_freq()    
#    update_ticks_format( ax['heat'].yaxis, scale_freq )
#    update_ticks_format( ax['plot'].xaxis, get_current_xscale() )
    xdata_freq = np.arange(0, num_freq, 1) * scale_freq
    print( xdata_freq[-1])        
    view_freq = get_view_freq()
    
    replot_shadow( [True, True])
    replot_light()
    reform_axis()
    
    replot_heat()

    ax['freq'].remove()
    ax['freq'], sval_freq = make_freq_slider()
    
#    reform_ticks()
    fig.canvas.draw_idle()

############################
# Replot support functions #
############################
def update_light():
    if plot_mode == 'freq_cut':
        update_line( l2, xdata_time, zdata.T[:,idx_freq] )
        update_color( l2, color_freq_light, alpha_freq_light )
        update_marker_color( l2, color_freq_light )
    else:
        update_line( l2, xdata_freq, zdata[:,idx_time] )
        update_color( l2, color_time_light, alpha_time_light )
        update_marker_color( l2, color_time_light )

#def update_shadow( replot_flags ):
#    for ilh in range(len(replot_flags) ):
#        if replot_flags[ilh]:
#            if plot_mode == 'freq_cut':
#                update_shadow_ilh( ilh, idx_freq, xdata_time, zdata.T )
#            else:
#                update_shadow_ilh( ilh, idx_time, xdata_freq, zdata )
def update_shadow( replot_flags ):
    for i, need_replot in enumerate(replot_flags):
        if need_replot:
            if plot_mode == 'freq_cut':
                update_shadow_ilh( i, idx_freq, xdata_time, zdata.T )
            else:
                update_shadow_ilh( i, idx_time, xdata_freq, zdata )

def update_shadow_ilh( ilh, index_light, xdata, ydata ):
    '''
    update line object substituting current xdata/ydata with new ones
    IN:  ilh         : 0:lower, 1:hiher
         index_light : index of main line (light)
    '''
    shadow_list = get_shadow_list( index_light, num_lower, num_higher )
    if shadow_list[ilh]:
        for ( idx, lo ) in zip( shadow_list[ilh], l2shadow[ilh] ):
#            print plot_mode, val, lo
            update_line( lo, xdata, ydata[:,idx] )

def replot_light( ):
    global l2
    if l2:
        l2.remove()
    if plot_mode == 'freq_cut':
        col = color_freq_light
        l2, = ax['plot'].plot( xdata_time, zdata[idx_freq,:], '-o',
            	color=col, markerfacecolor=col, markeredgecolor=col, lw=1, ms=2 )
    else:
        col = color_time_light
        l2, = ax['plot'].plot( xdata_freq, zdata[:,idx_time], '-o',
            	color=col, markerfacecolor=col, markeredgecolor=col, lw=1, ms=2 )

##??    ax['plot'].set_xscale(linlog)
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
                replot_shadow_ilh( ilh, idx_freq, xdata_time, zdata.T,
                                   color_freq_shadow[ilh], alpha_freq_shadow[ilh] * alpha_neighbors )
            else:
                replot_shadow_ilh( ilh, idx_time, xdata_freq, zdata,
                                   color_time_shadow[ilh], alpha_time_shadow[ilh] * alpha_neighbors )

def replot_shadow_ilh( ilh, val, xdata, ydata, color, alpha ):
    shadow_list = get_shadow_list( val, num_lower, num_higher )
    if shadow_list[ilh]:
        l2shadow[ilh] = ax['plot'].plot( xdata, ydata[:,shadow_list[ilh]], '-', color=color, lw=1, alpha=alpha )

def replot_heat():
    global lcuttime
    global lcutfreq
    global im    

    if im:
        im.remove()
        lcuttime.remove()
        lcutfreq.remove()
    
    print( xdata_freq[-1])        
    
    X,Y = np.meshgrid( xdata_time, xdata_freq )
    im = ax['heat'].pcolormesh(X, Y, zdata, snap=True, cmap=cmap_name )
    if plot_mode is 'freq_cut':
        alpha_vl = 0
        alpha_hl = alpha_hm
    else:
        alpha_vl = alpha_hm
        alpha_hl = 0
    lcuttime = ax['heat'].axvline( idx_time, color=color_time_light, alpha=alpha_vl )
    lcutfreq = ax['heat'].axhline( idx_freq * scale_freq, lw=1, color=color_freq_light, alpha=alpha_hl )
    ax['heat'].set_yscale(linlog)    
    ax['heat'].axis( [ xdata_time.min(), xdata_time.max(), xdata_freq.min(), xdata_freq.max() ] ) # Set min/max values for each axis

def update_line( lo, xdata, ydata ):
    lo.set_xdata( xdata )
    lo.set_ydata( ydata )

def update_color( lo, color, alpha ):
    lo.set_color( color )
    lo.set_alpha( alpha )

def update_marker_color( lo, color ):
    lo.set_markerfacecolor( color )
    lo.set_markeredgecolor( color )

#def update_ticks_freq():
#    """
#    Obsolete
#    """
#    if linlog == 'linear':
#        ticklabels = np.arange( 0, sf/2+2, 2000 ).astype(int)
#        ticks = ticklabels / scale_freq
#    else:
#        ticklabels = np.array( [0.1, 1.0, 10.0,100.0] )
#        ticks = ticklabels / scale_freq
#    return ticks, ticklabels

def update_ticks_format( axis, scale ):
    """
    Set ticks and labels
    In: axis  : ax.xaxis or ax.yaxis
        scale : actual value per bin
    """
    axis.set_major_formatter( ticker.FuncFormatter( lambda x,pos: "%d" % (x*scale+0.5)) )
    if linlog != 'linear':
        grid = np.arange(np.log10(sf/2))[2:]
        locs = 10**grid/scale_freq
        axis.set_major_locator(ticker.FixedLocator(locs))
        print("locs- ",locs, locs*scale_freq)
    else:
        if scale == 1.0: # freq_cut 
            multbin_major = 10
            multbin_minor = 1
        elif sf > 19000: #  time_cut 5kHz ticks
            multbin_major = 5000/scale
            multbin_minor = multbin_major / 5
        else:         # time_cut 1KHz ticks
            multbin_major = 1000/scale
            multbin_minor = multbin_major / 10
        axis.set_major_locator(ticker.MultipleLocator(multbin_major))
        axis.set_minor_locator(ticker.MultipleLocator(multbin_minor))

#def reform_ticks():
#    if plot_mode == 'time_cut':
#        ax['plot'].set_xticks( ticks_freq )
#        ax['plot'].set_xticklabels( ticklabels_freq )
#    else:
#        ax['plot'].set_xticks( np.arange(0,zdata.shape[1]+10,10))
#        ax['plot'].set_xticklabels( np.arange(0,zdata.shape[1]+10,10))
#    return

def reform_axis( ):
    global plot_mode    
    global linlog
    ax['plot'].set_xscale(linlog)
    if plot_mode == 'time_cut':    
        ax['plot'].axis( view_freq )
        ax['plot'].set_xlabel( 'Freq' )
    else:
        ax['plot'].axis( view_time )
        ax['plot'].set_xlabel( 'Time' )

def get_shadow_list( index_light, num_lower, num_higher ):
    index_lower =  range( max( 0, index_light - num_lower ), index_light )
    index_higher =  range( index_light+1, min( num_neighbors[plot_mode], index_light + 1 + num_higher ) )
    return  [ index_lower, index_higher ]

def get_replot_flag( index_light ):
    '''
    Check lower and higher shadow at least one exist and return True/False respectively
    '''
    shadow_list = get_shadow_list( index_light, num_lower, num_higher )
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
        val = idx_freq
        val_max = num_freq - 1
    else:
        val = idx_time
        val_max = num_time - 1
    num_lower = min( val, int(num_shadow_all/2) )
    num_higher = num_shadow_all - num_lower
    hami = num_higher - ( val_max - val  )
    if( hami > 0 ):
        num_lower += hami
        num_higher -= hami

def remake_zdata( zdata0, zscale='linear'):
    """
    Scale data along z-axis
    In:  zdata0 : input data   (freq,time)
         zscale : z-scale power|magnitude|linear
    Ret: zdata  : scaled zdata
         zmin,zmax : min or max og scaled zdata
    """    
    global zdata    
    if zscale in 'power':
        zdata = 10.0 * np.log10( zdata0 )
    elif zscale in 'magnitude':
        zdata = 20.0 * np.log10( zdata0 )
    else:
        zdata = zdata0
    zmin = np.min(zdata)    
    zmax = np.max(zdata)
    return zdata, zmin, zmax

def get_scale_freq():
    """
    Get frequency per bin
    """
#    return sf / 2 / (zdata.shape[0]-1)
    return sf / 2 / (num_freq-1)

def make_freq_slider():
    ax['freq'] = plt.axes([0.1, 0.05, 0.3, 0.03], facecolor=axcolor)
    sval_freq = Slider(ax['freq'], 'freq', 0.0, (num_freq-1) * scale_freq, valinit=idx_freq * scale_freq, valfmt='%.1f')
    sval_freq.on_changed(callback_freq_cut )
    return ax['freq'], sval_freq

def freq_to_idx( freq, scale ):
    return int( round( freq / scale ) )  # "%.1f" in Slidebar applies unexpected round ...

def get_view_freq():
    return [ 6, sf/2, 1, zmax*1.1 ]

####################
# Other Call Backs #
####################
def onclick_motion(event):
    if ( drag == True ) and ( ax['heat'] == event.inaxes ) :
        update_slider( event )

def onclick_press(event):
    global drag
    drag = True
    if ( ax['heat'] == event.inaxes ):
        update_slider( event )

def onclick_release(event):
    global drag
    drag = False

def on_key(event):
    """
    Modifi slider value if arrow key pressed
    """
    if event.key == 'right':
        print( 'on_key      ', stime.val+1)
        stime.set_val( min( stime.val+1, stime.valmax ) )
    if event.key == 'left':
        print( 'on_key      ', stime.val-1)
        stime.set_val( max( stime.val-1, stime.valmin ) )
    if event.key == 'up':
        sval_freq.set_val( min(sval_freq.val + scale_freq, sval_freq.valmax) )
    if event.key == 'down':
        sval_freq.set_val( max(sval_freq.val - scale_freq, sval_freq.valmin) )

def on_scroll(event):
    """
    Modify slider value if wheel scroled
    """
    if event.step > 0:
        if plot_mode == 'time_cut':
            stime.set_val( min( stime.val+1, stime.valmax ) )
        else:
            sval_freq.set_val( min(sval_freq.val + scale_freq, sval_freq.valmax) )
    else:
        if plot_mode == 'time_cut':
            stime.set_val( max( stime.val-1, stime.valmin ) )
        else:
            sval_freq.set_val( max(sval_freq.val - scale_freq, sval_freq.valmin) )

def update_slider( event ):
    if plot_mode == 'time_cut':
        print( 'update_slider ', event.xdata )
        stime.set_val( min(event.xdata, stime.valmax) )
    else:
        sval_freq.set_val( min(event.ydata, sval_freq.valmax) )

def reset(event):
    global plot_mode
    stime.reset()
    sval_freq.reset()
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
    ##############    
    # Parameter  #
    ##############
    sf = 11025
    plot_mode = 'time_cut'
    num_higher = 0
    num_lower = 0
    alpha_neighbors = 1.0
    linlog = 'linear'
#    yscale = [ 'linear', 'log']
    
    ############
    #   Data   #    
    ############
    # make test data
#    zdata = make_test_data()
    zdata0 = read_csv_file('wave2d.csv') # (time,freq)
#    zdata = read_freq_file('wave2d.csv')
    print( 'zdata.shape=', zdata0.shape )

    # Transpose Inpu Data
    zdata0 = zdata0.T # (freq,time)

    num_freq = zdata0.shape[0] # num of freq
    num_time = zdata0.shape[1] # num of time
    num_neighbors = { 'freq_cut':num_freq,     # virtical (y)axis
               'time_cut':num_time}      # holizontal (x)axis

    # Make notch For Debug    
    zdata0[:,20] /= 10.0
    zdata0[:,50] /= 10.0
    zdata0[160,:] /= 10.0
    zdata0[40,:] /= 10.0
    
    # Remake Input Data Pow/Mag/Linar
    zdata, zmin, zmax = remake_zdata( zdata0, zscale='linear')
#    zlim = [ max(zdata.ravel())*1.1, 10**( np.floor(np.log10(max(zdata.ravel()))) + 1 ) ]
#    zmax_log = 10**( np.floor(np.log10(zmax ) ) + 1 )
#    zmax_lin = np.max(zdata)*1.1

    scale_freq = get_scale_freq( )
    ###########################
    # time cut plot variables
    ###########################
    
#    color_time_light = 'blue'
#    color_time_light = 'dodgerblue'
    color_time_light = 'navy'
#    color_time_shadow = [ 'lightsteelblue', 'lightblue' ]
#    color_time_shadow = [ color_time_light, color_time_light ]
    color_time_shadow = [ 'dodgerblue', 'dodgerblue' ]
    alpha_time_light = 1.0
#    alpha_time_shadow = [ 0.3, 0.3]
    alpha_time_shadow = [ 1.0, 1.0]


#    view_freq = [ [0, num_freq, 1, zmax_lin ],
#                  [0, num_freq, 1, zmax_log ] ] # [xmin,xmax,ymin,ymax]
#    xdata_freq = np.arange(0.0, num_freq * scale_freq, scale_freq )
#    view_freq = [ 0, num_freq, 1, zmax*1.1 ]
    view_freq = get_view_freq()
    #xdata_freq = np.arange(0, num_freq, 1)
    xdata_freq = np.arange(0, num_freq, 1) * scale_freq

    ###########################
    # freq cut plot variables #
    ###########################
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
#    view_time = [ [0, num_time, 1, zmax_lin ],
#                  [0, num_time, 1, zmax_log ] ]
    view_time = [ 0, num_time, 1, zmax*1.1 ]
    xdata_time = np.arange(0, num_time, 1)

    ###########################
    # Heat Map variables
    ###########################
    idx_time = 0
    idx_freq = 0
    # position line
    alpha_hm = 1.0
    cmap_name='copper'
#    cmap_name='gray'
#    cmap_name='pink'

    ##############
    # Start Plot #
    ##############
    fig = plt.figure(figsize=[12,7])
#    plt.subplots_adjust( bottom=0.20 )

    ax = {}

    # Subplot 121
    ax['heat'] = plt.subplot(121)
    '''
#    X,Y = np.meshgrid( range(zdata.shape[1]+1), range(zdata.shape[0]+1) )
    X,Y = np.meshgrid( xdata_time, xdata_freq )
##    Y = Y * scale_freq
#    im = plt.imshow(zdata, aspect='auto', origin='lowier', interpolation='none', cmap=cm.get_cmap(name=cmap_name) )
#    im = plt.pcolormesh(X, Y, zdata, snap=True, cmap=cm.get_cmap(name=cmap_name) )
    im = plt.pcolormesh(X, Y, zdata, snap=True, cmap=cmap_name )

    plt.ylabel('freq')
    plt.xlabel('time')
    plt.yscale(linlog)    
    lcuttime = plt.axvline( idx_time, color=color_time_light, alpha=alpha_hm )
    lcutfreq = plt.axhline( idx_freq, lw=1, color=color_freq_light, alpha=alpha_hm )
    plt.axis( [ xdata_time.min(), xdata_time.max(), xdata_freq.min(), xdata_freq.max() ] ) # Set min/max values for each axis
#    plt.axis( [ X[0,:].min(), X[0,:].max(), Y[:,0].min(), Y[:,0].max() ] ) # Set min/max values for each axis
#    update_ticks_format( ax['heat'].yaxis, scale_freq )
    '''
    im = []
    replot_heat()    
    plt.ylabel('freq')
    plt.xlabel('time')

    # Subplot 122
    ax['plot'] = plt.subplot(122)
    l2shadow = [[],[]]
    replot_shadow( [True, True])
    l2 = []
    replot_light( )
    plt.grid(which='major',color='gray',linestyle='-')
    plt.grid(which='minor',color='gray',linestyle='--', alpha=0.3)
    plt.axis( view_freq )
    plt.ylabel('Pow/Mag')
    plt.xlabel('freq')
#    update_ticks_format( ax['plot'].xaxis, scale_freq )
#    ticks_freq, ticklabels_freq = update_ticks_freq()    
#    reform_ticks()

    # Slider
    axcolor = 'lightgoldenrodyellow'
#    ax['freq'] = plt.axes([0.1, 0.05, 0.3, 0.03], facecolor=axcolor)
    ax['freq'], sval_freq = make_freq_slider()
    ax['time'] = plt.axes([0.1, 0.10, 0.3, 0.03], facecolor=axcolor)
    ax['alpha'] = plt.axes([0.8, 0.05, 0.15, 0.03], facecolor=axcolor)
    ax['neighbors'] = plt.axes([0.8, 0.10, 0.15, 0.03], facecolor=axcolor)

    stime = Slider(ax['time'], 'time', 0, num_time-1, valinit=50+idx_time, valfmt='%d')
#    sval_freq = Slider(ax['freq'], 'freq', 0.0, (num_freq-1) * scale_freq, valinit=idx_freq * scale_freq, valfmt='%.1f')
    sneighbors = Slider(ax['neighbors'], '# neighbors', 0, num_time-1, valinit=num_lower, valfmt='%d')
    salpha = Slider(ax['alpha'], 'alpha', 0, 1, valinit=np.sqrt(alpha_neighbors), valfmt='%.2f')

    stime.on_changed(callback_time_cut)
    salpha.on_changed(callback_alpha)
    sneighbors.on_changed(callback_neighbors)

    # Reset button
#    resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
#    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
#    button.on_clicked(reset)

    # Linear/Log  button
    ax['linlog'] = plt.axes([0.48, 0.10, 0.05, 0.03])
    button_linlog = Button( ax['linlog'], 'Lin/Log', color=axcolor, hovercolor='0.975')
    button_linlog.on_clicked(callback_linlog)

    # TextBox for Sample Rate
    ax['sf'] = plt.axes([0.48, 0.05, 0.1, 0.03])
    textbox_sf = TextBox(ax['sf'], 'SF', initial=str(sf), color=axcolor, hovercolor='0.975' )
    textbox_sf.on_submit(callback_sf)

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
