# Python3
# Nino Carrillo
# 4 Mar 26

import sys
from scipy.io.wavfile import write as writewav
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft, fft, fftfreq
from scipy.signal import firwin

def PlotPilots(start_carrier, end_carrier, pilot_count):
	pilot_indicies = []
	carrier_interval = (end_carrier-start_carrier) // (pilot_count - 1)
	if carrier_interval % 2:
		carrier_interval -= 1
	this_pilot = start_carrier
	if this_pilot % 2:
		this_pilot += 1
	this_coord = 1+0j
	for i in range(pilot_count):
		if this_pilot <= end_carrier:
			pilot_indicies.append([int(this_pilot), this_coord])
		this_pilot += carrier_interval
		this_coord *= 1j
	return pilot_indicies

def GenSCPre(sym_n, pre_n, start_carrier, end_carrier, seed):
	np.random.seed(seed)
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

def GenSCWidePre(sym_n, pre_n, start_carrier, end_carrier, start_data_carrier, end_data_carrier, seed):
	np.random.seed(seed)
	baseband = np.zeros(sym_n, dtype='complex')
	# set low energy constellation points for the preamble
	coord = np.sqrt(2)/2
	# 
	for i in range(start_data_carrier, end_data_carrier+1):
		if i % 2:
			# i is odd, zero this subcarrier
			baseband[i] = 0
		else:
			# i is even
			baseband[i] = np.random.choice([-coord,coord]) + (np.random.choice([-coord,coord]) * 1j)
		# make output real by setting negative frequency subcarrier to conjugate
		if i > 0:
			baseband[(sym_n - i)] = baseband[i].conj()
	# Now add the extended guard carriers
	for i in range(start_carrier, start_data_carrier):
		if i % 2:
			# i is odd, zero this subcarrier
			baseband[i] = 0
		else:
			# i is even
			baseband[i] = np.random.choice([-coord,coord]) + (np.random.choice([-coord,coord]) * 1j)
		# make output real by setting negative frequency subcarrier to conjugate
		if i > 0:
			baseband[(sym_n - i)] = baseband[i].conj()
	for i in range(end_data_carrier+1, end_carrier+1):
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
	sc_audio *= sym_n / (end_data_carrier - start_data_carrier)
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

def GenConstellation(bits):
	constellation = []
	if bits == 1:
		constellation.append(1+0j)
		constellation.append(-1+0j)
	elif bits == 2:
		coord = np.sqrt(2)/2
		for x in range(-1,2,2):
			for y in range(-1,2,2):
				constellation.append(x*coord+y*coord*1j)
	elif bits == 3:
		angle = 2 * np.pi / 8
		for i in range(8):
			constellation.append(np.exp(1j*i*angle))
	elif bits == 41: # traditional qam-16
		coord = np.sqrt(2)/6
		for x in range(-3,4,2):
			for y in range(-3,4,2):
				constellation.append(x*coord+y*coord*1j)
	elif bits >= 4:
		# Try something different
		target_len = np.power(2,bits)
		len_c = 0
		step = 1
		rad = 0
		e = 0
		while (len_c != target_len) or (rad < .9) or (e < 0.5):
			x = -1
			y = -1
			x += np.random.rand() * step
			even_odd = 0
			constellation = []
			while y <= 1:
				new_point = x + (1j * y)
				#print(f'  {new_point:.3f}')
				x += step
				if x > 1:
					x = x - 2
					if even_odd % 2:
						x += step/2
					y += step
				even_odd += 1
				if np.abs(new_point) <= 1:
					constellation.append(new_point)
			len_c = len(constellation)
			if len_c < target_len:
				#step is too big
				step /= 2
			elif len_c > target_len:
				#step is too small
				step *= 1.5
			elif len_c  == target_len:
				step *= 1.5
			rad = np.max(np.abs(constellation))
			e = np.sum(np.power(np.abs(constellation),2)) / np.power(2,bits)
			#print(f'step: {step:.3f}, len: {len_c}, rad: {rad:.3f}')
		print(f'Bits: {bits}, Step: {step:.6f}, Rad: {rad:.3f}, e: {e:.3f}')

	# draw a unit circle
	circle_points = 1000
	circle_coords = np.zeros(circle_points, dtype='complex')
	
	for i in range(1000):
		circle_coords[i]=np.exp(2j*np.pi*i/circle_points)

	plt.figure(figsize=(6,6))
	plt.plot(circle_coords.real, circle_coords.imag, color='grey', linewidth=1)
	for coord in constellation:
		plt.scatter(coord.real, coord.imag, color='blue', s=5)
	plt.title(f'{bits}-bit QAM constellation')
	plt.ylim(-1,1)
	plt.xlim(-1,1)
	plt.show()
	return(constellation)

def GenRandomQAM(sym_n, pre_n, start_carrier, end_carrier, pilots, constellation):
	baseband = np.zeros(sym_n, dtype='complex')

	for i in range(start_carrier, end_carrier+1):
		baseband[i] = np.random.choice(constellation)
	# Add pilot carriers
	for pilot in pilots:
		baseband[pilot[0]] = pilot[1]
	# Make real by setting conjugates negative:
	for i in range(start_carrier, end_carrier+1):
		if i > 0:
			baseband[(sym_n - i)] = baseband[i].conj()
	# Generate the audio
	sc_audio = ifft(baseband, sym_n)
	# Scale the audio based on carriers in use
	sc_audio *= sym_n / (end_carrier - start_carrier)
	# prepend cyclic prefix
	sc_audio = np.concatenate([sc_audio[-pre_n:], sc_audio])
	return sc_audio.real

def GenRandomQPSK(sym_n, pre_n, start_carrier, end_carrier, pilots):
	baseband = np.zeros(sym_n, dtype='complex')
	coord = np.sqrt(2)/2
	for i in range(start_carrier, end_carrier+1):
		baseband[i] = np.random.choice([-coord,coord]) + (np.random.choice([-coord,coord]) * 1j)
	# Add pilot carriers
	for pilot in pilots:
		baseband[pilot[0]] = pilot[1]
	# Make real by setting conjugates negative:
	for i in range(start_carrier, end_carrier+1):
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
	if len(sys.argv) != 5:
		print("Incorrect arg count. Usage: python3 gen_ofdm_wav.py <dac bits> <qam bits> <repeat count> <output wav file>")
		sys.exit(2)

	dac_bits = int(sys.argv[1])
	qam_bits = int(sys.argv[2])
	repeat_n = int(sys.argv[3])
	wav_file_name = sys.argv[4]
	audio_sample_rate = 12000
	cp_n = 16
	fft_n = 512
	f_0 = 600
	f_max = 3030
	sc_guard_n = 3 # number of extra even bins on each side of spectrum in SC preamble
	pilot_n = 4
	sym_rate = audio_sample_rate / (fft_n + cp_n)
	bin_width = audio_sample_rate / fft_n
	bin_0 = int(np.ceil(f_0/bin_width))
	bin_max = int(np.floor(f_max/bin_width))
	sc_bin_0 = bin_0
	x_n = 0
	while x_n < sc_guard_n:
		sc_bin_0 -= 1
		if sc_bin_0 % 2 == 0:
			x_n += 1
	sc_bin_max = bin_max
	x_n = 0
	while x_n < sc_guard_n:
		sc_bin_max += 1
		if sc_bin_max % 2 == 0:
			x_n += 1
	sc_bin_n = (sc_bin_max - sc_bin_0) + 1
	bin_n = (bin_max - bin_0) + 1
	data_carrier_n = bin_n - pilot_n
	pilots = PlotPilots(bin_0, bin_max, pilot_n)


	print(f'Audio Sample Rate: {audio_sample_rate}')
	print(f'FFT N: {fft_n}')
	print(f'Bin Width: {bin_width}')
	print(f'Symbol Rate: {sym_rate:.2f}')
	print(f'Cyclic Prefix: {cp_n} samples, {1000*cp_n/audio_sample_rate:.2f} mS, {100*cp_n / (cp_n + fft_n):.1f}%')
	print(f'Schmidl Cox Bin 0: {sc_bin_0}, {sc_bin_0 * bin_width:.3f} Hz')
	print(f'Schmidle Cox Bin Max: {sc_bin_max}, {sc_bin_max * bin_width:.3f} Hz')
	print(f'Bin 0: {bin_0}, {bin_0 * bin_width:.3f} Hz')
	print(f'Bin Max: {bin_max}, {bin_max * bin_width:.3f} Hz')
	print(f'Occupied bin count: {bin_n}')
	print(f'Pilot count: {pilot_n}')
	print(f'Data subcarrier count: {bin_n-pilot_n}')
	print(f'QPSK Data/sym: {(bin_n-pilot_n)*2} bits, {(bin_n-pilot_n) /4:.2f} bytes')
	print(f'QPSK Bits/Sec: {data_carrier_n*sym_rate*2:.0f}')
	print(f'8PSK Bits/Sec: {data_carrier_n*sym_rate*3:.0f}')
	print(f'16QAM Bits/Sec: {data_carrier_n*sym_rate*4:.0f}')
	print(f'32QAM Bits/Sec: {data_carrier_n*sym_rate*5:.0f}')
	print(f'64QAM Bits/Sec: {data_carrier_n*sym_rate*6:.0f}')
	print(f'128QAM Bits/Sec: {data_carrier_n*sym_rate*7:.0f}')
	print(f'256QAM Bits/Sec: {data_carrier_n*sym_rate*8:.0f}')
	print(f'Pilots: {pilots}')

	
	# Generate Schmidl-Cox preamble
	audio_samples = np.zeros(fft_n+cp_n)
	audio_samples = np.concatenate([audio_samples,GenSCWidePre(fft_n, cp_n, sc_bin_0, sc_bin_max, bin_0, bin_max, 0)])
	constellation = GenConstellation(qam_bits)

	audio_samples = np.concatenate([audio_samples, GenRandomQAM(fft_n, cp_n, bin_0, bin_0+bin_n, pilots, constellation)])
	audio_samples = np.concatenate([audio_samples, GenRandomQAM(fft_n, cp_n, bin_0, bin_0+bin_n, pilots, constellation)])
	audio_samples = np.concatenate([audio_samples, GenRandomQAM(fft_n, cp_n, bin_0, bin_0+bin_n, pilots, constellation)])
	audio_samples = np.concatenate([audio_samples, GenRandomQAM(fft_n, cp_n, bin_0, bin_0+bin_n, pilots, constellation)])
	audio_samples = np.concatenate([audio_samples, GenRandomQAM(fft_n, cp_n, bin_0, bin_0+bin_n, pilots, constellation)])
	audio_samples = np.concatenate([audio_samples, GenRandomQAM(fft_n, cp_n, bin_0, bin_0+bin_n, pilots, constellation)])
	audio_samples = np.concatenate([audio_samples, GenRandomQAM(fft_n, cp_n, bin_0, bin_0+bin_n, pilots, constellation)])
	audio_samples = np.concatenate([audio_samples, GenRandomQAM(fft_n, cp_n, bin_0, bin_0+bin_n, pilots, constellation)])
	audio_samples = np.concatenate([audio_samples, np.zeros(fft_n+cp_n)])
	

	audio_samples = np.tile(audio_samples, repeat_n)

	
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