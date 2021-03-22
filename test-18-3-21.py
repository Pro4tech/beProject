import numpy as np
import adi
import matplotlib.pyplot as plt
import time
import math
from scipy import signal
sample_rate = 350e5 # Hz
threshold=35000
##Tx Test
#sample_rate=1e6
#center_freq=915e6

def filter(b,a,freq):
		z1=signal.lfilter_zi(b,a)
		pl=signal.lfilter(b,a,freq)
		return(pl)

def setup(center_freq,sample_rate):
	def power(val):
		p=abs(val**2)
		pdb=math.log10(p)
		return(pdb)
	power=np.vectorize(power)
	name = center_freq/1e9
	sdr = adi.Pluto("ip:192.168.2.1")
	sdr.sample_rate = int(sample_rate)
	sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
	sdr.rx_lo = int(center_freq)
	sdr.rx_buffer_size = 1024 # this is the buffer the Pluto uses to buffer samples
	#print("center_frq:{:.3f}".format(name))
	samples = sdr.rx()
	freqx=np.linspace(int(-sample_rate//2+center_freq),int(sample_rate//2+center_freq),int(1024))
	frq=np.fft.fft(samples)
	freq=np.fft.fftshift(frq)
	#freq=power(freq)
	sig_avg = abs(sum(filter(b,a,freq)))
	print("Signal Avg:",sig_avg)
	if sig_avg > threshold :
		print("Active Band:{}Ghz".format(name))
	#freq1=power(freq)
	plot=filter(b,a,freq)
	plt.plot(freqx,plot)
	plt.title("Frequency Amplitude") #for Single TimeFrame
	plt.show()
	#plt.savefig("{:.3f}Ghz-FA.png".format(name))
	#plt.close()	
	samplegroup=[]
	for _ in range(1000):
		samples=sdr.rx()
		frq=np.fft.fft(samples)
		freq=np.fft.fftshift(frq)
		#freq=power(freq)
		samplegroup.append(abs(freq))
	plt.imshow(samplegroup)
	plt.set_cmap('hot')
	plt.title("Waterfall Diagram") #for 1000ms TimeFrame
	plt.savefig("{:.3f}Ghz-WD.png".format(name))
	plt.close()
	#plt.show()

b,a =signal.butter(3,.75)
print("Sample rate:",sample_rate)
for center_freq in range(2465,2515,50):
	center_freq=center_freq*1e6
	print("\n center_freq:",center_freq)
	setup(center_freq,sample_rate)
print("End of Code")

