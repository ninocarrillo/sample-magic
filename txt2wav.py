# Python3
# Nino Carrillo
# 4 Mar 26

import sys
from scipy.io.wavfile import write as writewav
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
		print("Incorrect arg count. Usage: python3 txt2wav.py <input txt file> <input sample rate> <interpolation rate> <carrier freq> <repeat count> <output wav file>")
		sys.exit(2)

	baseband_sample_rate = float(sys.argv[2])
	interpolation_rate = int(sys.argv[3])
	carrier_freq = float(sys.argv[4])
	repeat_count = int(sys.argv[5])
	wav_file_name = sys.argv[6]
	audio_sample_rate = baseband_sample_rate * interpolation_rate

	# Open the input text file and read data into a list of complex float values
	baseband_sample_list = []
	with open(sys.argv[1], encoding="utf-8") as input_file:
		input_file_lines = input_file.readlines()
		# The file contains one I/Q sample per line, I and Q separated by space
		for line in input_file_lines:
			split_line = line.split()
			baseband_sample_list.append([float(split_line[0]), float(split_line[1])])

	# Create complex baseband sample values from the list value pairs
	baseband_samples = np.zeros(len(baseband_sample_list), dtype=complex)
	for i in range(len(baseband_sample_list)):
		baseband_samples[i] = baseband_sample_list[i][0] + (baseband_sample_list[i][1] * 1j)

	# Create expanded baseband by inserting zeros
	expanded_baseband_samples = np.zeros(len(baseband_samples) * interpolation_rate, dtype=complex)
	for i in range(len(baseband_sample_list)):
		expanded_baseband_samples[i * interpolation_rate] = baseband_samples[i]

	# Repeat these samples
	expanded_baseband_samples = np.tile(expanded_baseband_samples, repeat_count)

	# Apply interpolation filter
	interp_fir = firwin(2047, [baseband_sample_rate / 2], pass_zero='lowpass', fs=audio_sample_rate)
	expanded_baseband_samples = np.convolve(expanded_baseband_samples, interp_fir)
	expanded_baseband_sample_count = len(expanded_baseband_samples)

	audio_samples = np.zeros(expanded_baseband_sample_count, dtype=complex)
	# Mix complex samples with sine wave at carrier freq to translate spectrum
	for i in range(expanded_baseband_sample_count):
		time_var = 2 * np.pi * i * carrier_freq / audio_sample_rate
		audio_samples[i] = expanded_baseband_samples[i] * np.exp(time_var * 1j)

	# Discard complex samples
	audio_samples = audio_samples.real

	# Scale audio samples
	audio_samples = audio_samples * interpolation_rate

	# Perform FFT analysis of modulation stages
	baseband_psd = AnalyzeSpectrum(baseband_samples, baseband_sample_rate, 0.99)
	audio_psd = AnalyzeSpectrum(audio_samples, audio_sample_rate, 0.99)
	expanded_baseband_psd = AnalyzeSpectrum(expanded_baseband_samples, audio_sample_rate, 0.99)

	fig, ax = plt.subplots(2,3)
	fig.tight_layout()
	plt.subplot(231)
	plt.plot(baseband_samples.real, baseband_samples.imag, '.', ms=2)
	plt.title('Baseband Samples')
	plt.xlabel('in-phase')
	plt.ylabel('quadrature')
	plt.xlim(-0.5,0.5)
	plt.ylim(-0.5,0.5)
	plt.grid(True)
	plt.subplot(232)
	plt.plot(baseband_psd[0], baseband_psd[1], '.', ms=2)
	plt.xlim(-baseband_sample_rate, baseband_sample_rate)
	plt.ylim(-60,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Baseband Spectrum")
	plt.grid(True)
	plt.subplot(233)
	plt.plot(expanded_baseband_psd[0], expanded_baseband_psd[1], '.', ms=2)
	plt.xlim(-3000, 3000)
	plt.ylim(-100,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Interpolated Baseband Spectrum")
	plt.grid(True)
	plt.subplot(234)
	plt.plot(audio_samples, linewidth=1)
	plt.title("Audio Samples")
	plt.ylim(-0.5,0.5)
	plt.subplot(235)
	plt.plot(audio_psd[0], audio_psd[1], '.', ms=2)
	plt.xlim(0, 3000)
	plt.ylim(-100,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Audio spectrum")
	plt.grid(True)
	plt.show()


	# Write wavfile out
	writewav(wav_file_name, int(audio_sample_rate), audio_samples)
			

if __name__ == "__main__":
	main()