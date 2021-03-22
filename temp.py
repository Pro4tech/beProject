import numpy as np
import adi
import matplotlib.pyplot as plt
import time
import math

i=0
sample_rate = 350e5 # Hz
Center_freq = [2465,2565,2665,2765,2865,2965] #2.475*10^9 Hz
for center_freq in Center_freq
	i=i+1
	center_freq=center_freq*1e6
	print("center_freq:",center_freq)
	##Tx Test
	#sample_rate=1e6
	#center_freq=915e6
	#Normalize Graph Power
	def power(val):
		p=abs(val**2)
		pdb=math.log10(p)
		return(pdb)
	power=np.vectorize(power)
	sdr = adi.Pluto("ip:192.168.2.1")
	sdr.sample_rate = int(sample_rate)
	sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
	sdr.rx_lo = int(center_freq)
	sdr.rx_buffer_size = 1024 # this is the buffer the Pluto uses to buffer samples
	#samples = sdr.rx() # receive samples off Pluto
	##SDR Gain Mod
	#sdr.gain_control_mode_chan0 = "manual" # turn off AGC
	#gain = 50.0 # allowable range is 0 to 74.5 dB
	#sdr.rx_hardwaregain_chan0 = gain # set receive gain
	print("Sample_rate:",sample_rate)
	print(" center_frq:",center_freq)
	#for Single Freq
	samples = sdr.rx()
	freqx=np.linspace(int(-sample_rate//2+center_freq),int(sample_rate//2+center_freq),int(1024))
	print("Samples:",samples)
	frq=np.fft.fft(samples)
	freq=np.fft.fftshift(frq)
	freq=power(freq)
	plt.plot(freqx,abs(freq))
	plt.title("Frequency Amplitude")
	#plt.show()
	plt.savefig("{}-FA.png".format(i))
	samplegroup=[]
	for _ in range(1000):
		samples=sdr.rx()
		frq=np.fft.fft(samples)
		freq=np.fft.fftshift(frq)
		freq=power(freq)
		samplegroup.append(abs(freq))
	plt.imshow(samplegroup)
	plt.set_cmap('hot')
	plt.title("Waterfall Diagram")
	plt.savefig("{}-WD.png".format(i))
	#plt.show()


