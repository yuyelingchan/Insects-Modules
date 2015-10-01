import matplotlib.pyplot as plt
import numpy


TITLE_FONT= {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'bold',
              'verticalalignment':'bottom'}


# frequency-time spectrum
def freq_spec(fspec, title, rate, time=30, fig_size=(8,4)):

    (im,axes) = plt.subplots(1,1,figsize=fig_size)
    
    axes.pcolormesh(numpy.linspace(0, time, len(fspec[0])+1), \
                    numpy.linspace(0, rate/2000, len(fspec)), \
                    fspec)
    
    axes.set_title(title, **TITLE_FONT)
    
    axes.set_ylim(0, rate/2000)
    axes.set_xlim(0, time)
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
        
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    
    return im
   
       
# power spectrum
def power_freq(pspec, title, rate, fig_size=(8,4)):
    
    (im,axes) = plt.subplots(1,1,figsize=fig_size)
    
    freq = numpy.fft.rfftfreq(len(pspec[0]))
    freq_arr = numpy.tile((abs(freq * rate)),(len(pspec),1))
    
    axes.plot(freq_arr,pspec)
    axes.set_title(title, **TITLE_FONT)
    axes.set_xlabel("Frequency/Hz")
  
    

# Plot logged/unlogged mel spectrum after applying filterbank
def filtered_mel(mspec, title, fig_size = (8,4)):
    
    (im, axes) = plt.subplots(1,1,figsize = fig_size)
    axes.imshow(numpy.transpose(mspec), cmap="jet", origin="lower", aspect="auto", interpolation="nearest")
    axes.set_title(title, **TITLE_FONT)
    axes.set_xlabel("Time(s)")

    
    
# DCT coefficient cepstrum
def dct_ceps(dceps, title, time=30, fig_size=(8,4)):
    
    (im, axes) = plt.subplots(1, 1, figsize=fig_size)
    axes.imshow(numpy.transpose(dceps), origin="lower", aspect="auto",interpolation="nearest",\
                extent=[0, time, 0, len(dceps[0])])
    
    axes.set_xlabel("Time(s)")
    axes.set_title(title, **TITLE_FONT)
    
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
        
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
        
        
    return im


        
# Plot filters
def filters(filters, rate, fig_size=(8,4)):
    
    (im, axes) = plt.subplots(1, 1, figsize=fig_size)
    
    freq_range = numpy.fft.rfftfreq((len(filters[0])-1)*2)
    filter_freq = numpy.tile((abs(freq_range*rate)),(len(filters),1))
    
    for i in range(0, len(filters)):
        axes.plot(filter_freq[i],filters[i])
        
    axes.set_xlabel("Frequency(Hz)")
    axes.set_xlim(0,rate/2)

    
        
#Plot modulation coefficient
def mod(mod, title, min_freq=0, max_freq=344, rate=44100):
    
    (im_mod,axes_mod) = plt.subplots(1,1,figsize=(8,4))
    
    axes_mod.imshow(mod.T, aspect="auto", origin="lower", interpolation="nearest",\
                    extent=[min_freq, max_freq, 0, rate/2000])
    
    for tick in axes_mod.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
        
    for tick in axes_mod.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    axes_mod.set_title(title, **TITLE_FONT)
    axes_mod.set_xlabel("Frequency(Hz)" )
    axes_mod.set_ylabel("Frequency(kHz)")
    
    return im_mod



#Plot first 10 percent coefficients
def mod10p(mod, title, min_freq=0, max_freq=344, rate=44100):
    
    ten_percent = int(numpy.floor(len(mod)*0.1))

    (im_mod10p, axes_mod10p) = plt.subplots(1,1,figsize=(8,4))
    axes_mod10p.imshow(mod[:ten_percent,:].T, aspect="auto", origin="lower",\
                       interpolation="nearest", extent=[min_freq, (max_freq*0.1), 0, rate/2000])
    
    for tick in axes_mod10p.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
        
    for tick in axes_mod10p.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
                
    axes_mod10p.set_title(title, **TITLE_FONT)
    axes_mod10p.set_xlabel("Frequency(Hz)")
    axes_mod10p.set_ylabel("Frequency(kHz)")
    
    return im_mod10p



#Plot only the first, mean, std or max mod
def mod1(coe_arr, title, rate = 44100): 
    (im_mod1, axes_mod1) = plt.subplots(1,1,figsize=(8,4))
    axes_mod1.plot(numpy.linspace(0, rate/2000, len(coe_arr)), coe_arr)
    
    axes_mod1.set_xlim(0, rate/2000)
    for tick in axes_mod1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
        
    for tick in axes_mod1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    
    axes_mod1.set_title(title, **TITLE_FONT)
    axes_mod1.set_xlabel("Frequency(Hz)")
    
    return im_mod1



#Plot log modulation coefficients
def logmod(log_mod, title, rate=44100, nfft=64):
    xscale = numpy.power(10.0, numpy.linspace(numpy.log10(1/60.0),numpy.log10(344.0), 49))
    
    (im_logmod, axes_logmod) = plt.subplots(1, 1, figsize=(10,5)) 

    lmod = numpy.array(log_mod).T

    axes_logmod.pcolormesh(numpy.linspace(0, len(lmod.T), len(lmod.T)+1), \
                           numpy.linspace(0, rate/2000, nfft/2+1), lmod)

    axes_logmod.set_xlim(0, len(lmod.T))
    axes_logmod.set_xticks(numpy.linspace(0, len(lmod.T), 13))

    x = []
    for i in range(0, 13):
        x.append(round(xscale[i*4],1))
    
    axes_logmod.set_xticklabels(x)   
    axes_logmod.set_ylim(0, rate/2000)

    for tick in axes_logmod.xaxis.get_major_ticks():
        tick.label.set_fontsize(10) 
        
    for tick in axes_logmod.yaxis.get_major_ticks():
        tick.label.set_fontsize(10)
        
    axes_logmod.set_title(title, **TITLE_FONT)
    axes_logmod.set_xlabel("Frequency(Hz)")
    axes_logmod.set_ylabel("Frequency(kHz)")
        
    return im_logmod



#Plot mel spectrum
def mel_spec(mspec, title, signal, rate=44100):
    
    (im_m,axes_m) = plt.subplots(1,1,figsize=(8,4))
    axes_m.pcolormesh(numpy.linspace(0, len(signal)/rate, len(mspec[0])), \
                      numpy.linspace(0, len(mspec), len(mspec)), numpy.array(mspec))
    
    axes_m.set_xlim(0, len(signal)/rate)
    for tick in axes_m.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
        
    for tick in axes_m.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
                
    axes_m.set_title(title, **TITLE_FONT)
    axes_m.set_xlabel("Time(s)")
    axes_m.set_ylabel("Mel")
    
    return im_m


#Plot mel mod
def mel_mod(mmod, title, min_freq=0, max_freq=344):

    (im_mod,axes_mod) = plt.subplots(1,1,figsize=(8,4))
    
    axes_mod.imshow(mmod.T, aspect="auto", origin="lower", interpolation="nearest",\
                    extent=[min_freq, max_freq, 0, len(mmod[0])])
    
    for tick in axes_mod.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
        
    for tick in axes_mod.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
        
    axes_mod.set_title(title, **TITLE_FONT)
    axes_mod.set_xlabel("Frequency(Hz)")
    axes_mod.set_ylabel("Mel")
    
    return im_mod


#Plot mel logmod
def mel_logmod(mlogmod, title, xscale, rate=44100, nfft=64):
    
    (im_logmod, axes_logmod) = plt.subplots(1, 1, figsize=(10,5)) 

    lmod = numpy.array(mlogmod).T

    axes_logmod.pcolormesh(numpy.linspace(0, len(lmod.T), len(lmod.T)+1), \
                           numpy.linspace(0, len(lmod), len(lmod)+1), lmod)

    axes_logmod.set_xlim(0, len(lmod.T))
    axes_logmod.set_xticks(numpy.linspace(0, len(lmod.T), 13))

    x = []
    for i in range(0, 13):
        x.append(xscale[i*4])
    
    axes_logmod.set_xticklabels(x)   
    
    for tick in axes_logmod.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
        
    for tick in axes_logmod.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
        
    axes_logmod.set_title(title, **TITLE_FONT)
    axes_logmod.set_xlabel("Frequency(Hz)")
    axes_logmod.set_ylabel("Mel")
    
    return im_logmod


#Plot dct logmod
def dct_logmod(dlogmod, title, nbin=48, rate=44100, nfft=64):
    
    bin_freqs = numpy.power(10.0, numpy.linspace(numpy.log10(1/60), numpy.log10(344), nbin))
    freqs = numpy.around(bin_freqs, 4)
    xscale = numpy.append(xscale, numpy.array([344.5167]))
        
    (im_logmod, axes_logmod) = plt.subplots(1, 1, figsize=(10,5)) 

    lmod = numpy.array(dlogmod).T

    axes_logmod.pcolormesh(numpy.linspace(0, len(lmod.T), len(lmod.T)+1), \
                           numpy.linspace(0, len(lmod), len(lmod)+1), lmod)
    
    axes_logmod.set_xlim(0, len(lmod.T))
    axes_logmod.set_xticks(numpy.linspace(0, len(lmod.T), 13))

    x = []
    for i in range(0, 7):
        x.append(xscale[i*8])
    
    axes_logmod.set_xticklabels(x)   

    for tick in axes_logmod.xaxis.get_major_ticks():
        tick.label.set_fontsize(14) 
        
    for tick in axes_logmod.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
        
    axes_logmod.set_title(title, **TITLE_FONT)
    axes_logmod.set_xlabel("Frequency(Hz)")
        
    return im_logmod

    