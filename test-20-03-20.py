# Author:Pritesh Naik
# BE ETC 171104067

import numpy as np
import adi
import matplotlib.pyplot as plt
import time
import math 
from scipy import signal
import pandas as pd
sample_rate = 350e5 # Hz
b,a =signal.butter(3,.75)
final = np.array([])


def filter(b,a,freq):
		z1=signal.lfilter_zi(b,a)
		pl=signal.lfilter(b,a,freq)
		return(pl)

def setup(center_freq,sample_rate,final):
	def power(val):
		p=abs(val**2)
		pdb=math.log10(p)
		return(pdb)
	##SDR Setup Commands
	power=np.vectorize(power)
	fname = center_freq/1e9 
	name=str('{:.3f}Ghz'.format(fname))
	sdr = adi.Pluto("ip:192.168.2.1")
	sdr.sample_rate = int(sample_rate)
	sdr.rx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
	sdr.rx_lo = int(center_freq)
	sdr.rx_buffer_size = 1024 # this is the buffer the Pluto uses to buffer samples
	
	##Sampling Signal at Carrier Freq
	samples = sdr.rx()
	freqx=np.linspace(int(-sample_rate//2+center_freq),int(sample_rate//2+center_freq),int(1024))
	frq=np.fft.fft(samples)
	freq=np.fft.fftshift(frq)
	
	sig_avg = signal.find_peaks(filter(b,a,freq),height=10000) 	#Detecting Required Amplitude Peaks
	print('Size:',sig_avg[0].size)								#if(tuple!=empty)->then Adjust OffsetCorrection->then append in final[]
	if(sig_avg[0].size>0):
		print("Active Band:{}Ghz".format(name))
		temp=np.array(sig_avg[0])
		print("Signal Avg:",temp)
		for i in range(0,temp.size):
			temp[i]=temp[i]*(sample_rate/1000)+center_freq
		print("Signal Avg:",temp)
		final=np.concatenate((final, temp))
	
	##Storing Dataframe in .csv format
	store=pd.DataFrame(freq)									# Storing Dataframe using PANDAS -> IMPORT TO TX IN PANDAS
	store.to_csv(name+"-Dataframe.png")
	store=pd.DataFrame(sig_avg[0])									# Storing Dataframe using PANDAS -> IMPORT TO TX IN PANDAS
	store.to_csv(name+"-Peaks_Dataframe.png")
	
	##Plotting Frequency Response			#freq1=power(freq)
	plt.plot(freqx,freq)						#plt.plot(freq)
	plt.xlim((-10,1030))
	plt.ylim((-10000,55000))
	plt.xlabel('Frequency(Base Freq+Offset*10)Hz')
	plt.ylabel('Amplitude of Signal')
	plt.title("Frequency Amplitude:"+name) #for Single TimeFrame
	plt.savefig(name+"-FA.png")				#plt.show()
	plt.close()	
	samplegroup=[]						#Ploting Waterfall Diagram
	for _ in range(1000):
		samples=sdr.rx()
		frq=np.fft.fft(samples)
		freq=np.fft.fftshift(frq)				#freq=power(freq)
		samplegroup.append(abs(freq))
	plt.imshow(samplegroup)
	plt.set_cmap('hot')
	plt.xlabel('Frequency(Base Freq+offset*10)Hz')
	plt.title("Waterfall Diagram:"+name) #for 1000ms TimeFrame
	plt.savefig(name+"-WD.png") 	#plt.show()
	plt.close()
	return final
	
#Main Function Codes
print("Sample rate:",sample_rate)
for center_freq in range(2465,2715,50):
	center_freq=center_freq*1e6
	print("\n center_freq:",center_freq)
	final1=setup(center_freq,sample_rate,final)
	final=final1
print("Final Array:",final)
print("End of Code")

#Outputs:
# Sample rate: 35000000.0
#  center_freq: 2465000000.0
# Signal Avg: [75 78]

#  center_freq: 2515000000.0
# Signal Avg: []

#  center_freq: 2565000000.0
# Signal Avg: []

#  center_freq: 2615000000.0
# Signal Avg: []

#  center_freq: 2665000000.0
# Signal Avg: [557 561]

# End of Code


