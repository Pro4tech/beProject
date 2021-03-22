import numpy as np
import adi
import time
sample_rate = 550e3 # Hz
srt_freq = 91.7e6 # Hz
end_freq=2.5e9
sdr = adi.Pluto("ip:192.168.2.1")
time.sleep(0.1)
for i in range(int(srt_freq),int(end_freq),int(sample_rate)):
 print("running at:",i)
 sdr.sample_rate = int(sample_rate)
 sdr.tx_rf_bandwidth = int(sample_rate) # filter cutoff, just set it to the same as sample rate
 sdr.tx_lo = int(i)
 sdr.tx_hardwaregain = 0 # set the transmit gain to the lowest transmit power.  LOWER this value to make it transmitter higher, yes it's backwards for some reason, -90 is the lowest you can go
 N = 10000 # number of samples to transmit at once
 t = np.arange(N)/sample_rate
 samples = 0.5*np.exp(2.0j*np.pi*100e3*t) # Simulate a sinusoid of 100 kHz, so it should show up at 915.1 MHz at the receiver
 samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs

#Transmit our batch of samples 100 times, so it should be 1 second worth of samples total, if USB can keep up
 for i in range(2000):
  sdr.tx(samples) # transmit the batch of samples once
  #time.sleep(0.1) #include a wait time of 1 sec
print("END")
