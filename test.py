# # # Author:Pritesh Naik
# # # BE ETC 171104067

#Importing Neccesary Libraries
import numpy as np
import adi
import matplotlib.pyplot as plt
from matplotlib import transforms
import time
import math 
from scipy import signal
import pandas as pd

#Program Initilisation
#Sample Min:300e3 Max:500e5
sample_rate = 250e4 # Hz
b,a =signal.butter(3,.2)
final = np.array([])
flist=np.array([])

##Function Declaration for Low pass filter
def filter(b,a,freq):
		z1=signal.lfilter_zi(b,a)
		pl=signal.lfilter(b,a,freq)
		return(pl)

##Main Code Function Declaration
def setup(center_freq,sample_rate,final):
	#PSD_Formatting Function
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
	
	#Detection of Required Amplitude Peaks
	sig_avg = signal.find_peaks(filter(b,a,freq),height=10000) 	
	print('Size:',sig_avg[0].size)								#if(tuple!=empty)->then Adjust OffsetCorrection->then append in final[]
	temp=np.array(sig_avg[0])
	if(sig_avg[0].size>0):
		print("Active Band:{}".format(name))
		for i in range(0,temp.size):
			temp[i]=temp[i]*(sample_rate/1000)+center_freq
		print("Signal Peaks:",temp)
		final=np.concatenate((final, temp))
	
	#Plotting Frequency Response			#freq1=power(freq)
	plt.plot(freqx,freq)
	#plt.xlim((-10,1030))
	#plt.ylim((-10000,55000))
	plt.xlabel('Frequency(Base Freq+Offset)Hz')
	plt.ylabel('Amplitude of Signal')
	plt.title("Frequency Amplitude:"+name) #for Single TimeFrame
	plt.savefig(name+"-FA.png")				#/////
	plt.show()
	plt.close()	
	
	#Storing Dataframe in .csv format
	store=pd.DataFrame(freq)									# Storing Dataframe using PANDAS -> IMPORT TO TX IN PANDAS
	store.to_csv(name+"-Dataframe.csv")
	
	# #issue:Relative plot need abs plot spectogram(Pending Work)
	# samplegroup=[]						#Ploting Waterfall Diagram
	# for _ in range(1000):
	# 	samples=sdr.rx()
	# # 	frq=np.fft.fft(samples)
	# # 	freq=np.fft.fftshift(frq)				#freq=power(freq)
	# 	samplegroup.append(abs(samples))
	# plt.imshow(samplegroup)
	# plt.set_cmap('hot')
	# plt.xlabel('Frequency(Base Freq+offset)Hz')
	# plt.title("Waterfall Diagram:"+name) #for 1000ms TimeFrame
	#samples=sdr.rx()
	f, t, Sxx = signal.spectrogram(samples, sample_rate,scaling='spectrum', axis=- 1, mode='psd')
	base=plt.gca().transData
	rot=transforms.Affine2D().rotate_deg(-90)
	plt.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Sxx, axes=0), shading='gouraud',transform=rot+bas
	plt.xlabel('Frequency [Hz]')
	plt.ylabel('Time [sec]')
	#plt.show()
	plt.savefig(name+"-WD.png") 	#///
	# plt.show()
	plt.close()
	return final
	
##Main Function Codes
print("Sample rate:",sample_rate,'Hz')
rangeF=[2475,2435,]
for center_freq in range(2405,2515,3):
	center_freq=int(center_freq*1e6)
	print("Center_freq:",center_freq,'Hz')
	final1=setup(center_freq,sample_rate,final)
	final=final1
print("Final Peaks_Array:",final)

#Storing Dataframe in .csv format		     
store=pd.DataFrame(final)									# Storing Dataframe using PANDAS -> IMPORT TO TX IN PANDAS
store.to_csv("Peak-Dataframe.csv")
print("End of Code")
