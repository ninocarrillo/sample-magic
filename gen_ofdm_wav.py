# Python3
# Nino Carrillo
# 4 Mar 26

import sys
from scipy.io.wavfile import write as writewav
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft, fft, fftfreq
from scipy.signal import firwin

def GenSCPre(sym_n, pre_n, start_carrier, end_carrier):
	baseband = np.zeros(sym_n, dtype='complex')
	# set low energy constellation points for the preamble
	coord = np.sqrt(2)/2
	for i in range(start_carrier, end_carrier+1):
		if i % 2:
			# i is odd, zero this subcarrier
			baseband[i] = 0
		else:
			# i is even
			baseband[i] = np.random.choice([-coord,coord]) + (np.random.choice([-coord,coord]) * 1j)
		# make output real by setting negative frequency subcarrier to conjugate
		if i > 0:
			baseband[(sym_n - i)] = baseband[i].conj()
	# Generate the audio
	sc_audio = ifft(baseband, sym_n)
	# Scale the audio based on carriers in use
	sc_audio *= sym_n / (end_carrier - start_carrier)
	# prepend cyclic prefix
	sc_audio = np.concatenate([sc_audio[-pre_n:], sc_audio])
	return sc_audio.real
	
def GenProbe(sym_n, pre_n, start_carrier, end_carrier):
	baseband = np.zeros(sym_n, dtype='complex')
	coord = np.sqrt(2)/2
	for i in range(start_carrier, end_carrier+1):
		baseband[i] = np.random.choice([-coord,coord]) + (np.random.choice([-coord,coord]) * 1j)
		if i > 0:
			baseband[(sym_n - i)] = baseband[i].conj()
	# Generate the audio
	sc_audio = ifft(baseband, sym_n)
	# Scale the audio based on carriers in use
	sc_audio *= sym_n / (end_carrier - start_carrier)
	# prepend cyclic prefix
	sc_audio = np.concatenate([sc_audio[-pre_n:], sc_audio])
	return sc_audio.real
		
	

def AnalyzeSpectrum(waveform, sample_rate, power_ratio):
	fft_n = len(waveform)
	time_step = 1 / sample_rate
	x = np.linspace(0.0, fft_n * time_step, fft_n, endpoint = False)
	x_fft = fftfreq(fft_n, time_step)
	waveform_fft = fft(waveform)
	fft_max = max(abs(waveform_fft))
	waveform_fft = waveform_fft / fft_max

	# create power spectrum
	waveform_psd = np.zeros(len(waveform_fft))
	for i in range(len(waveform_fft)):
		waveform_psd[i] = pow(abs(waveform_fft[i]),2)

	# calculate power bandwidth
	# first determine total
	total_power = 0
	for sample in waveform_psd:
		total_power += sample
	# now sum from center of spectrum out
	total_power = total_power / 2

	power_sum = 0
	i = -1
	while (power_sum < power_ratio) and (i < len(waveform_psd)):
		i += 1
		power_sum += waveform_psd[i] / total_power
	obw = x_fft[i] * 2

	obw_mask = np.zeros(4)
	obw_x = np.zeros(4)
	obw_mask[0] = 10*np.log10(waveform_psd[i])
	obw_mask[1] = 0
	obw_mask[2] = 0
	obw_mask[3] = 10*np.log10(waveform_psd[i])
	obw_x[0] = -x_fft[i]
	obw_x[1] = -x_fft[i]
	obw_x[2] = x_fft[i]
	obw_x[3] = x_fft[i]
	return([x_fft, 10*np.log10(waveform_psd), obw_x, obw_mask, obw])

def main():
	# check correct version of Python
	if sys.version_info < (3, 0):
		print("Python version should be 3.x, exiting")
		sys.exit(1)
	# check correct number of parameters were passed to command line
	if len(sys.argv) != 4:
		print("Incorrect arg count. Usage: python3 gen_ofdm_wav.py <dac bits> <symbol count> <output wav file>")
		sys.exit(2)

	dac_bits = int(sys.argv[1])
	symbol_count = int(sys.argv[2])
	wav_file_name = sys.argv[3]
	audio_sample_rate = 48000
	cp_n = 32
	fft_n = 2048
	
	# Generate Schmidl-Cox preamble
	
	audio_samples = np.zeros(fft_n+cp_n)
	audio_samples = np.concatenate([audio_samples, GenSCPre(fft_n, cp_n, 13, 13+116)])
	audio_samples = np.concatenate([audio_samples, GenProbe(fft_n, cp_n, 13, 13+116)])
	audio_samples = np.concatenate([audio_samples, GenProbe(fft_n, cp_n, 13, 13+116)])
	audio_samples = np.concatenate([audio_samples, GenProbe(fft_n, cp_n, 13, 13+116)])
	audio_samples = np.concatenate([audio_samples, GenProbe(fft_n, cp_n, 13, 13+116)])
	audio_samples = np.concatenate([audio_samples, GenProbe(fft_n, cp_n, 13, 13+116)])
	
	plt.figure()
	plt.plot(audio_samples.real)
	plt.plot(audio_samples.imag)
	plt.legend(['real','imag'])
	plt.title('Audio Samples')
	plt.show()


	# Perform FFT analysis of modulation stages
	audio_psd = AnalyzeSpectrum(audio_samples, audio_sample_rate, 0.99)

	fig, ax = plt.subplots(1,2)
	fig.tight_layout()
	plt.subplot(121)
	plt.plot(audio_samples, linewidth=1)
	plt.title("Audio Samples")
	plt.ylim(-0.5,0.5)
	plt.subplot(122)
	plt.plot(audio_psd[0], audio_psd[1], '.', ms=2)
	plt.xlim(-4000, 4000)
	plt.ylim(-100,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Audio spectrum")
	plt.grid(True)
	plt.show()
	
	# convert audio samples to integer real values
	
	audio_samples = audio_samples.real
	audio_samples *= (np.power(2,dac_bits-1) - 1)
	audio_samples = np.round(audio_samples,0)

	# Write wavfile out
	writewav(wav_file_name, int(audio_sample_rate), audio_samples.astype(np.int16))
			

if __name__ == "__main__":
	main()