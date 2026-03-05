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
	if len(sys.argv) != 5:
		print("Incorrect arg count. Usage: python3 wav2txt.py <input wav file> <decimation rate> <carrier freq> <output txt file>")
		sys.exit(2)

	decimation_rate = int(sys.argv[2])
	carrier_freq = float(sys.argv[3])

	try:
		audio_sample_rate, audio_samples  = readwav(sys.argv[1])
	except:
		print('Unable to open audio file.')
		sys.exit(3)

	audio_sample_count = len(audio_samples)

	# Mix audio with negative complex carrier freq to move spectrum to baseband
	baseband_samples = np.zeros(len(audio_samples), dtype=complex)
	for i in range(audio_sample_count):
		baseband_samples[i] = audio_samples[i] * np.exp(2j * np.pi * i * (-carrier_freq) / audio_sample_rate)

	# Decimate baseband samples:
	baseband_samples = baseband_samples[::decimation_rate]
	baseband_sample_rate = audio_sample_rate / decimation_rate

	# Normalize baseband samples
	baseband_samples = baseband_samples / np.max(np.abs(baseband_samples))

	# Perform FFT of final audio signal
	audio_psd = AnalyzeSpectrum(audio_samples, audio_sample_rate, 0.99)
	baseband_psd = AnalyzeSpectrum(baseband_samples, baseband_sample_rate, 0.99)

	plt.figure()
	plt.subplot(223)
	plt.plot(baseband_samples.real, '.')
	plt.plot(baseband_samples.imag, '.')
	plt.title('Baseband Samples')
	plt.subplot(224)
	plt.plot(baseband_psd[0], baseband_psd[1])
	plt.xlim(-baseband_sample_rate, baseband_sample_rate)
	plt.ylim(-100,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Baseband Spectrum")
	plt.grid(True)
	plt.subplot(221)
	plt.plot(audio_samples, '.')
	plt.title("Audio Samples")
	plt.subplot(222)
	plt.plot(audio_psd[0], audio_psd[1])
	plt.xlim(0, 3000)
	plt.ylim(-100,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Audio Spectrum")
	plt.grid(True)
	plt.show()

	try:
		with open(sys.argv[4], 'w', encoding='utf-8') as output_file:
			for complex_sample_pair in baseband_samples:
				print(f'{complex_sample_pair.real:.6f} {complex_sample_pair.imag:.6f}', file=output_file)
	except:
		printf(f'Unable to create output file {sys.argv[4]}')

			

if __name__ == "__main__":
	main()