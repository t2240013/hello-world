import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.widgets import Slider, Button, TextBox, RadioButtons
import simpleaudio as sa

#############################
###  Global Variables     ###
#############################
# Const
FS = 44100
GAIN_MAX = 1.0
GAIN_MIN = 0.001

# Variable
Freq = 1000
FreqStep = 10
GaindB = 0.0
GainLn0 = 1.0
Duration = 0.5
PlayCh = np.array( [ 1, 0 ] )
PlayObj = None

#############################
###  Callback Functions (Widget)  ###
#############################
def start_stop(event):
    if is_playing():
        stop_playing()
    else:
        start_playing()

def set_0dB(event):
    global GaindB
    global GainLn0
    print( 'GaindB= ', GaindB, ' GainLn0= ', GainLn0 )
    gain = GainLn0 * dB2Gain( GaindB )
    if gain < 0.001:
        print( 'Error: too small gain')
    elif gain > 1.0:
        print( 'Error: too big gain')
    else:
        GainLn0 = gain
        GaindB = 0.0
        textbox_gain.set_val(str(GaindB))
    print( GainLn0 )

def set_freq(text):
    global Freq
    Freq = eval(text)

def set_gain(text):
    global GaindB
    GaindB = eval(text)

def set_duration(text):
    global Duration
    Duration = eval(text)
    if Duration > 100:
        Duration = 100
    textbox_duration.set_val(str(Duration))

def dB_up(event):
    global GaindB
    GaindB += 1
    textbox_gain.set_val(str(GaindB))
    restart_playing()

def dB_down(event):
    global GaindB
    GaindB -= 1
    textbox_gain.set_val(str(GaindB))
    restart_playing()

def select_ch(label):
    global PlayCh
    if label == 'L':
        PlayCh = [ 1, 0]
    elif label == 'R':
        PlayCh = [ 0, 1]
    else:
        PlayCh = [ 1, 1]
    print( PlayCh )

def freq_step(label):
    global FreqStep
    stepdic = { 'x10':10, 'x100':100, 'x1000':1000}
    FreqStep = stepdic[label]
    print( 'FreqStep=', FreqStep)

####################
# Callback Functions (Key/Mouse) #
####################
def onclick_motion(event):
    #print( 'motion')
    return

def onclick_press(event):
    #print( 'press')
    return

def onclick_release(event):
    #print( 'release')
    return

def on_key(event):
    #print( event.key )
    if event.key == " ":
        restart_playing()

def on_scroll(event):
    global Freq
    global GaindB
    print( event.step )
    if ax['freq'] == event.inaxes:
        Freq += event.step * FreqStep
        Freq = dual_clip( Freq, 0, FS/2 )
        textbox_Hz.set_val(str(Freq))
        textbox_Hz.text_disp.set_size(15)
    if ax['gain'] == event.inaxes:
        GaindB += event.step
        textbox_gain.set_val(str(GaindB))
        textbox_gain.text_disp.set_size(15)
    #restart_playing()

############################
# Support Functions #
############################
def dB2Gain(val):
    return np.power( 10.0, val/20.0 )

def mkpee( gain, dur, lr, fx, fs ):
    winlen = 1500
    win = np.hanning(winlen*2)
    win = win[:winlen]

    dw = 2 * np.pi * fx / fs
    ww = np.arange( 0, dur*fs) * dw
    yy = gain * np.sin( ww )
    yy[:winlen] *= win # fade in
    yy[-winlen:] *= win[::-1] #fade out

    yyyy = np.concatenate( ( yy.reshape(-1,1), yy.reshape(-1,1) ), axis=1) # LR pee-
    zzzz = np.zeros_like( yyyy )[:fs//2,:] # LR pause
    sec = np.concatenate( ( yyyy, zzzz ), axis=0 ) * lr
    secs = np.tile( sec, (3,1) ) * 32767
    return secs.astype(np.int16)

def is_playing():
    if PlayObj == None:
        print( 'PlayObj==None')
        return False
    else:
        if PlayObj.is_playing():
            print( 'PlayObj is playing')
            return True
        else:
            print( 'PlayObj is stoped')
            return False

def stop_playing():
    global Playing
    sa.stop_all()

def start_playing():
    global PlayObj
    gain = GainLn0 * dB2Gain( GaindB)
    if ( gain < GAIN_MIN ) or ( gain > GAIN_MAX ):
        print( 'invalid gain= ', gain)
    else:
        data = mkpee( gain, Duration, PlayCh, Freq, FS )
        print( 'start: gain=%f (%d) freq=%d max=%d' % (gain, GaindB, Freq, np.max(data)) )
        PlayObj = sa.play_buffer( data, 2, 2, FS )

def restart_playing():
    if is_playing():
        stop_playing()
    start_playing()

def dual_clip(x, min, max ):
    if x < min:
        x = min
    elif x > max:
        x = max
    return x

###########################
###        Main         ###
###########################
if __name__ == '__main__':
    ##############
    # Parameter  #
    ##############
    axcolor = 'lightgoldenrodyellow'

    ############
    #   Data   #
    ############

    ##############
    # Start Plot #
    ##############
    fig = plt.figure(figsize=[8,4])
#    plt.subplots_adjust( bottom=0.20 )

    ax = {}

    # Set 0dB button
    ax['set_0dB'] = plt.axes([0.8, 0.8, 0.1, 0.1])
    button_set_0dB = Button(ax['set_0dB'], 'Set 0dB', color=axcolor, hovercolor='0.975')
    button_set_0dB.on_clicked(set_0dB)

    # Start/Stop button
    ax['start'] = plt.axes([0.55, 0.45, 0.15, 0.1]) # [left, buttom, width, hight ]
    button_start = Button(ax['start'], 'Start/Stop', color=axcolor, hovercolor='0.975')
    button_start.on_clicked(start_stop)

    # Freq Step Select
    ax['freq_step'] = plt.axes([0.55, 0.6, 0.2, 0.3], facecolor=axcolor)
    radio_freq_step = RadioButtons(ax['freq_step'], ('x10','x100','x1000' ) )
    radio_freq_step.on_clicked(freq_step)

    # TextBox for Hz
    ax['freq'] = plt.axes([0.2, 0.65, 0.3, 0.25])
    textbox_Hz = TextBox(ax['freq'], 'Freq(Hz) ', initial=str(Freq), color=axcolor, hovercolor='0.975' )
    textbox_Hz.label.set_size(20)
    textbox_Hz.text_disp.set_size(15)
    textbox_Hz.on_submit(set_freq)

    # TextBox duration (sec)
    ax['duration'] = plt.axes([0.2, 0.5, 0.3, 0.1])
    textbox_duration = TextBox(ax['duration'], 'Duration(sec) ', initial=str(Duration), color=axcolor, hovercolor='0.975' )
    textbox_duration.label.set_size(20)
    textbox_duration.text_disp.set_size(15)
    textbox_duration.on_submit(set_duration)

    # TextBox for Gain
    ax['gain'] = plt.axes([0.2, 0.15, 0.3, 0.25])
    textbox_gain = TextBox(ax['gain'], 'Gain(dB) ', initial=str(GaindB), color=axcolor, hovercolor='0.975' )
    textbox_gain.label.set_size(20)
    textbox_gain.text_disp.set_size(15)
    textbox_gain.on_submit(set_gain)

    # dB Up button
    ax['dB_up'] = plt.axes([0.55, 0.3, 0.1, 0.1])
    button_dB_up = Button(ax['dB_up'], '+', color=axcolor, hovercolor='0.975')
    button_dB_up.on_clicked(dB_up)

    # dB Down button
    ax['dB_down'] = plt.axes([0.55, 0.15, 0.1, 0.1])
    button_dB_down = Button(ax['dB_down'], '-', color=axcolor, hovercolor='0.975')
    button_dB_down.on_clicked(dB_down)

    # L,R L+R Select
    ax['LR'] = plt.axes([0.8, 0.15, 0.15, 0.3], facecolor=axcolor)
    radio_LR = RadioButtons(ax['LR'], ('L','R','LR' ) )
    radio_LR.on_clicked(select_ch)

    # Mouse
    drag = False
    cid_press = fig.canvas.mpl_connect('button_press_event', onclick_press)
    cid_release = fig.canvas.mpl_connect('button_release_event', onclick_release)
    cid_motion = fig.canvas.mpl_connect('motion_notify_event', onclick_motion)
    cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
    cid_scroll = fig.canvas.mpl_connect('scroll_event', on_scroll)

#    plt.tight_layout( w_pad = 2.0 )
#    plt.subplots_adjust( bottom=0.22, wspace=0.2, left=0.04, right=0.98, top=0.98 )
#    plt.subplots_adjust( bottom=0.22, wspace=0.2, left=0.065, right=0.98, top=0.98 )
    plt.show()
