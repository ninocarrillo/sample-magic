# Python3
# Nino Carrillo
# 4 Mar 26

import sys
from scipy.io.wavfile import write as writewav
from scipy.io.wavfile import read as readwav
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import firwin



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
	if len(sys.argv) != 7:
		print("Incorrect arg count. Usage: python3 wav2txt.py <input wav file> <decimation rate> <deci fir len> <sample offset> <carrier freq> <output txt file>")
		sys.exit(2)

	decimation_rate = int(sys.argv[2])
	deci_fir_len = int(sys.argv[3])
	sample_offset = int(sys.argv[4])
	carrier_freq = float(sys.argv[5])

	try:
		audio_sample_rate, audio_samples  = readwav(sys.argv[1])
	except:
		print('Unable to open audio file.')
		sys.exit(3)

	if (np.max(np.abs(audio_samples))) > 1:
		audio_samples = audio_samples / 32768

	print(f'Max audio amplitude: {np.max(np.abs(audio_samples)):.3f}')

	audio_sample_count = len(audio_samples)

	# Mix audio with complex carrier freq to move spectrum to baseband
	baseband_samples = np.zeros(audio_sample_count, dtype=complex)
	for i in range(audio_sample_count):
		time_var = 2 * np.pi * i * (-carrier_freq) / audio_sample_rate
		baseband_samples[i] = audio_samples[i] * np.exp(time_var * 1j)

	baseband_sample_rate = audio_sample_rate

	# Low-pass filter the complex baseband
	deci_fir = firwin(deci_fir_len, [1500], pass_zero='lowpass', fs=audio_sample_rate)
	baseband_samples = np.convolve(baseband_samples, deci_fir, mode='full')
	
	# Discard delayed samples
	baseband_samples = baseband_samples[int(((deci_fir_len - 1) / 2) - sample_offset):]
	pre_deci_baseband_sample_count = len(baseband_samples)
	
	pre_deci_baseband_samples = baseband_samples.copy()
	pre_deci_index = list(range(pre_deci_baseband_sample_count))
	# plt.figure()
	# plt.plot(pre_deci_index, pre_deci_baseband_samples.real, marker='o')

	# Decimate baseband samples:
	baseband_samples = baseband_samples[::decimation_rate]
	baseband_sample_rate = baseband_sample_rate / decimation_rate

	baseband_index = list(range(0,pre_deci_baseband_sample_count, decimation_rate))
	# plt.plot(baseband_index, baseband_samples.real, marker='x')
	# plt.show()

	# Perform FFT of final audio signal
	audio_psd = AnalyzeSpectrum(audio_samples, audio_sample_rate, 0.99)
	baseband_psd = AnalyzeSpectrum(baseband_samples, baseband_sample_rate, 0.99)

	fig, ax = plt.subplots(2,3)
	fig.tight_layout()
	plt.subplot(235)
	plt.plot(baseband_samples.real, baseband_samples.imag, '.', ms=2)
	plt.title('Baseband Samples')
	plt.xlabel('in-phase')
	plt.ylabel('quadrature')
	plt.xlim(-0.5,0.5)
	plt.ylim(-0.5,0.5)
	plt.grid(True)
	plt.subplot(234)
	plt.plot(baseband_psd[0], baseband_psd[1], '.', ms=2)
	plt.xlim(-baseband_sample_rate, baseband_sample_rate)
	plt.ylim(-60,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Baseband Spectrum")
	plt.grid(True)
	plt.subplot(231)
	plt.plot(audio_samples, linewidth=1)
	plt.title("Audio Samples")
	plt.ylim(-0.5,0.5)
	plt.subplot(232)
	plt.plot(audio_psd[0], audio_psd[1], '.', ms=2)
	plt.xlim(0, 3000)
	plt.ylim(-100,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Audio spectrum")
	plt.grid(True)
	plt.show()

	try:
		with open(sys.argv[6], 'w', encoding='utf-8') as output_file:
			for complex_sample_pair in baseband_samples:
				print(f'{complex_sample_pair.real:.6f} {complex_sample_pair.imag:.6f}', file=output_file)
	except:
		printf(f'Unable to create output file {sys.argv[6]}')

	# Attempt Schmidl-Cox conjugate correlation
	FFT_N = 64
	Oversample = int(6 / decimation_rate)
	L = int(Oversample * FFT_N / 2)
	
	# Create empty list to place metric P values
	d_range = len(baseband_samples) - (FFT_N * Oversample)
	P1 = np.zeros(d_range, dtype='complex')
	for d in range(d_range):
		P1[d] = 0 + 0j
		for m in range(L):
			P1[d] += baseband_samples[d + m].conj()*baseband_samples[d + m + L]

	# Create moving average filter with length equal to cyclic prefix sample count
	CP_N = 8
	MA_FIR = np.ones(CP_N * Oversample)
	P1_MA = np.convolve(P1, MA_FIR, mode='full') / (CP_N * Oversample)

	# Calculate the receive energy, R
	R = np.zeros(d_range)
	for d in range(d_range):
		R[d] = 0
		for m in range(L):
			R[d] += np.power(np.abs(baseband_samples[d + m + L]), 2) / Oversample

	# Normalize P1 
	P1_Norm = np.zeros(d_range)
	for d in range(d_range):
		x = np.power(R[d],2)
		if R[d] > 0.1:
			P1_Norm[d] = np.power(np.abs(P1_MA[d]), 2) / x


	plt.figure()
	plt.plot(np.abs(baseband_samples))
	plt.plot(P1.real)
	plt.plot(P1_MA.real)
	plt.plot(R)
	plt.plot(P1_Norm)
	plt.legend(['Baseband','P','P Moving Avg','R Energy','M Final Metric'])
	plt.show()


	# Decimate from selected sample
	pilot_index_1 = 7
	pilot_index_2 = 21
	pilot_index_3 = 43
	pilot_index_4 = 57
	fudge = -int(np.ceil(Oversample / 2))
	try:
		CP_Length = 8 * Oversample
		SC_Peak_Sample = 915
		#SC_Peak_Sample = 305
		#SC_Peak_Sample = 153
		SC_Offset = (2 * L) + CP_Length + fudge
		Start_i = SC_Peak_Sample + SC_Offset
		Symbol_Baseband = baseband_samples[Start_i:Start_i + (Oversample * FFT_N):Oversample]
		Symbol_Output = np.fft.fft(Symbol_Baseband, FFT_N)
		plt.figure()
		plt.title(f'Start Index {Start_i}, Oversample {Oversample}, Fudge {fudge}')
		plt.scatter(Symbol_Output[0:7].real, Symbol_Output[0:7].imag)
		#plt.scatter(Symbol_Output[pilot_index_1].real, Symbol_Output[pilot_index_1].imag)
		plt.scatter(Symbol_Output[22:43].real, Symbol_Output[22:43].imag)
		#plt.scatter(Symbol_Output[pilot_index_2].real, Symbol_Output[pilot_index_2].imag)
		plt.scatter(Symbol_Output[44:57].real, Symbol_Output[44:57].imag)
		#plt.scatter(Symbol_Output[pilot_index_3].real, Symbol_Output[pilot_index_3].imag)
		plt.scatter(Symbol_Output[58:64].real, Symbol_Output[58:64].imag)
		#plt.scatter(Symbol_Output[pilot_index_4].real, Symbol_Output[pilot_index_4].imag)
		#plt.scatter(Symbol_Output.real, Symbol_Output.imag)
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.grid('true')
		plt.show()
	except:
		pass

if __name__ == "__main__":
	main()