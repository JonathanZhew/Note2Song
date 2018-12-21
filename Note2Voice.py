
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from IPython.display import Audio
import time
import sys
from numpy import *
import scipy.io.wavfile
import wave

def load_wav(filename):
    try:
        wavedata=scipy.io.wavfile.read(filename)
        samplerate=int(wavedata[0])
        smp=wavedata[1]*(1.0/32768.0)
        if len(smp.shape)>1: #convert to mono
            smp=(smp[:,0]+smp[:,1])*0.5
        return (samplerate,smp)
    except:
        print ("Error loading wav: "+filename)
        return None

def paulstretch(samplerate, smp, stretch = 2, windowsize_seconds = 0.25):
    out_smp = []
    # make sure that windowsize is even and larger than 16
    windowsize = int(windowsize_seconds * samplerate)
    if windowsize < 16:
        windowsize = 16
    windowsize = int(windowsize / 2) * 2
    half_windowsize = int(windowsize / 2)

    # correct the end of the smp
    end_size = int(samplerate * 0.05)
    if end_size < 16:
        end_size = 16
    smp[len(smp) - end_size:len(smp)] *= linspace(1, 0, end_size)

    # compute the displacement inside the input file
    start_pos = 0.0
    displace_pos = (windowsize * 0.5) / stretch

    # create Hann window
    window = 0.5 - cos(arange(windowsize, dtype='float') * 2.0 * pi / (windowsize - 1)) * 0.5

    old_windowed_buf = zeros(windowsize)
    hinv_sqrt2 = (1 + sqrt(0.5)) * 0.5
    hinv_buf = hinv_sqrt2 - (1.0 - hinv_sqrt2) * cos(
        arange(half_windowsize, dtype='float') * 2.0 * pi / half_windowsize)

    while True:

        # get the windowed buffer
        istart_pos = int(floor(start_pos))
        buf = smp[istart_pos:istart_pos + windowsize]
        if len(buf) < windowsize:
            buf = append(buf, zeros(windowsize - len(buf)))
        buf = buf * window

        # get the amplitudes of the frequency components and discard the phases
        freqs = abs(fft.rfft(buf))

        # randomize the phases by multiplication with a random complex number with modulus=1
        ph = random.uniform(0, 2 * pi, len(freqs)) * 1j
        freqs = freqs * exp(ph)

        # do the inverse FFT
        buf = fft.irfft(freqs)

        # window again the output buffer
        buf *= window

        # overlap-add the output
        output = buf[0:half_windowsize] + old_windowed_buf[half_windowsize:windowsize]
        old_windowed_buf = buf

        # remove the resulted amplitude modulation
        output *= hinv_buf

        # clamp the values to -1..1
        output[output > 1.0] = 1.0
        output[output < -1.0] = -1.0

        # write the output to wav file
        out_smp = out_smp + list(int16(output * 32767.0))
        #outfile.writeframes(int16(output * 32767.0).tostring())

        start_pos += displace_pos
        if start_pos >= len(smp):
            #print("100 %")
            break
        #sys.stdout.write("%d %% \r" % int(100.0 * start_pos / len(smp)))
        #sys.stdout.flush()

    #outfile.close()
    return out_smp
#get the voice of do re mi

keys = []
fs,data = load_wav("note/c.wav")
keys.append(data)
fs,data = load_wav("note/d.wav")
keys.append(data)
fs,data = load_wav("note/e.wav")
keys.append(data)
fs,data = load_wav("note/f.wav")
keys.append(data)
fs,data = load_wav("note/g.wav")
keys.append(data)
fs,data = load_wav("note/a.wav")
keys.append(data)
fs,data = load_wav("note/b.wav")
keys.append(data)


#read the note
song5_123 = (
  ('c', 4), ('d', 4),('e', 4),('c', 4), ('c', 4), ('d', 4),('e', 4),('c', 4)
)

jinglebells = (
  ('e', 4), ('e', 4),('e', 8),('e', 4), ('e', 4),('e', 8),('e', 4), ('g', 4), ('c', 4),('d', 4),('e', 16),
  ('f', 4), ('f', 4),('f', 4),('f', 4), ('f', 4),('e', 4),('e', 4), ('e', 4),
  ('e', 4), ('d', 4),('d', 4),('e', 4),('d', 8),('g', 8),
  ('e', 4), ('e', 4),('e', 8),('e', 4), ('e', 4),('e', 8),('e', 4), ('g', 4), ('c', 4),('d', 4),('e', 16),
  ('f', 4), ('f', 4),('f', 4),('f', 4), ('f', 4),('e', 4),('e', 4), ('e', 4),
  ('g', 4), ('g', 4),('f', 4),('d', 4), ('c', 4)
)
tunes = ['c','d','e','f','g','a','b']

song = []
beat = 0
for note, value in jinglebells:
    index = int(tunes.index(note))
    voice = paulstretch(fs, keys[index], stretch=value/4)
    song = song + list(voice)
    beat = beat + value
    if beat >=16:
        beat = 0
        song = song + list(np.zeros(5000))

#print(wave, len(wave))
#plt.plot(wave)
#plt.show()
song = np.array(song)
#sd.play(song, fs)
#time.sleep(10)
#print(song)
#write the output to wav file

outfile = wave.open("out.wav", "wb")
outfile.setsampwidth(2)
outfile.setframerate(fs)
outfile.setnchannels(1)
outfile.writeframes(int16(song).tostring())
outfile.close()
