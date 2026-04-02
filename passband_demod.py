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
from scipy.signal import savgol_filter


def UpdateEqPilots(symbol, eq, pilots):
	pilot_correction = 1
	p_errors = np.zeros(len(pilots))
	for i in range(len(pilots)):
		p_errors[i] = (((np.angle(symbol[pilots[i][0]]) - np.angle(pilots[i][1]*pilot_correction)) + np.pi/2) % np.pi) - (np.pi/2)
		# Normalize phase error to frequency bin:
		p_errors[i] = p_errors[i] / pilots[i][0]
	print(f'Normalized pilot errors: ')
	for e in p_errors:
		print(f'{e*180/np.pi:.3f} deg/bin')
	e_avg = np.average(p_errors)
	# Create an array of phase corrections for each bin:
	phase_correction = np.zeros(len(eq))
	feedback_gain = 0.175
	#feedback_gain = 0
	for i in range(len(eq)):
		phase_correction[i] = e_avg * i * feedback_gain
	print(f'Avg pilot error: {e_avg*180/np.pi:.3f} deg/bin')
	# Adjust the equalizer phase:
	p = np.exp(1j*phase_correction)
	#eq *= p.conj()
	return p.conj()


def CalcEq(preamble_fft, ref_fft):
	# Calculate equalizer taps from the FFT of Schnmidl-Cox preamble symbol
	# Preamble symbol only has data in even bins, which means we will need
	# to interpolate equalizer taps for the odd bins. It also means we can
	# estimate signal-to-noise ratio by averaging the energy in the even bins
	# and the energy in the odd bins and computing a ratio.

	sig_e = 0
	noise_e = 0
	# Initially set the equalizer to the dot product of the conjugate of the 
	# reference symbol with the received symbol.
	# This computes bins beyond those occupied in the waveform, because compute
	# is cheap on my pc. Could make it more efficient in hardware by only doing
	# the calculations on the bins that contain the waveform.
	eq_taps = ref_fft.conj() * preamble_fft
	for i in range(len(eq_taps)): # step thru every equalizer tap
		if i % 2:
			# Odd subcarrier, only noise here
			# Compute the noise energy and sum it
			noise_e += np.power(np.abs(preamble_fft[i]),2)
			# now zero this equalizer bin, prep for interpolation
			eq_taps[i] = 0
		else:
			# Even subcarrier, signal here
			sig_e += np.power(np.abs(preamble_fft[i]),2)
			if np.abs(eq_taps[i]) > 0: # avoid divide-by-zero
				eq_taps[i] = eq_taps[i] / np.power(np.abs(eq_taps[i]),2)
			else: # in lieu of divide-by-zero, make this bin unity
				eq_taps[i] = 1
	if noise_e > 0: # avoid divide-by-zero
		snr = sig_e / noise_e
	else: # set some maximum snr
		snr = 1e9
	# Interpolate the empty equalizer bins.
	# The interpolation filter computes the average of the two surrounding bins for
	# every empty bin.
	# This is a 2-dimensional linear interpolation of the I/Q data. This works because
	# we have a very high number of carriers. In other words, the radial distance between
	# the equalizer taps is very low, so we can ignore the magnitude error caused by this
	# approach. If we interpolated along a curve between the two I/Q points, the angle of
	# the interpolated point would be the same but the magnitude would be slightly higher.
	# The magnitude error is approximately the inverse of the cosine of half the angle of
	# separation between the two surrounding occupied equalizer taps.
	interp_filter = np.array([0.5,1,0.5])
	#interp_filter = np.array([1/3,1/2,1/3,1/2,1/3])
	# Do the filter convolution
	eq_taps = np.convolve(eq_taps, interp_filter, mode='full')
	# Remove the filter delay:
	# Divide filter length by two and floor the result
	offset = len(interp_filter) // 2
	# Discard delayed samples to re-align equalizer taps to bins
	eq_taps = eq_taps[offset:-offset]
	# Do another pass with a moving average filter
	interp_filter = np.ones(3)/3
	eq_taps = np.convolve(eq_taps, interp_filter, mode='full')
	offset = len(interp_filter) // 2
	eq_taps = eq_taps[offset:-offset]
	# Return the complete equalizer, as an array of complex tap values. Also return 
	# linear SNR. 
	# Apply the equalizer by performing dot product to the received symbol (FFT output).
	return eq_taps.conj(), snr

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


def GenSCWidePreBB(sym_n, pre_n, start_carrier, end_carrier, start_data_carrier, end_data_carrier, seed):
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

	return baseband

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
		print("Incorrect arg count. Usage: python3 passband_demod.py <input wav file> <cfo> <fft n> <cp n> <bin 0> <bin max>")
		sys.exit(2)

	carrier_freq = float(sys.argv[2])
	fft_n = int(sys.argv[3])
	cp_n = int(sys.argv[4])
	bin_0 = int(sys.argv[5])
	bin_max = int(sys.argv[6])
	bin_n = (bin_max-bin_0) + 1


	sc_guard_n = 3 # number of extra even bins on each side of spectrum in SC preamble
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


	try:
		audio_sample_rate, audio_samples  = readwav(sys.argv[1])
	except:
		print('Unable to open audio file.')
		sys.exit(3)

	# Normalize audio amplitude
	audio_samples = audio_samples / max(np.abs(audio_samples))

	audio_sample_count = len(audio_samples)
	
	sym_rate = audio_sample_rate / (fft_n + cp_n)
	bin_width = audio_sample_rate / fft_n

	print(f'Audio Sample Rate: {audio_sample_rate}')
	print(f'FFT N: {fft_n}')
	print(f'Bin Width: {bin_width}')
	print(f'Symbol Rate: {sym_rate:.2f}')
	print(f'Cyclic Prefix time: {1000*cp_n/audio_sample_rate:.2f} mS, {100*cp_n / (cp_n + fft_n):.1f}%')


	pilot_n = 4
	pilots = PlotPilots(bin_0, bin_max, pilot_n)
	#pilots = PlotPilots(int(round(1500/bin_width)), int(round(2500/bin_width)), pilot_n)

	freq = np.fft.fftfreq(fft_n, 1/audio_sample_rate)

	# Mix audio with complex baseband carrier to correct LO freq only
	baseband_samples = np.zeros(audio_sample_count, dtype=complex)
	carrier_phase = 0
	print(f'Carrier Frequency Offset: {carrier_freq}')
	for i in range(audio_sample_count):
		time_var = 2 * np.pi * i * (-carrier_freq) / audio_sample_rate
		baseband_samples[i] = audio_samples[i] * np.exp((time_var + carrier_phase) * 1j)

	baseband_sample_rate = audio_sample_rate

	baseband_index = list(range(0,len(baseband_samples)))

	# Perform FFT of final audio signal
	audio_psd = AnalyzeSpectrum(audio_samples, audio_sample_rate, 0.99)
	baseband_psd = AnalyzeSpectrum(baseband_samples, baseband_sample_rate, 0.99)

	# Attempt Schmidl-Cox conjugate correlation
	
	L = int(fft_n / 2)

	SC_Scale = 10 / L
	
	# Create empty list to place metric P values
	d_range = len(baseband_samples) - fft_n
	P1 = np.zeros(d_range, dtype='complex')
	for d in range(d_range):
		P1[d] = 0
		for m in range(L):
			P1[d] += baseband_samples[d + m].conj()*baseband_samples[d + m + L]
			
	# Discard imaginary part of P1
	P1 = P1.real

	# Scale P1 to account for sample rate
	P1 *= SC_Scale

	# Create moving average filter with length equal to cyclic prefix sample count

	MA_FIR = np.ones(cp_n)
	P1_MA = np.convolve(P1, MA_FIR, mode='full') / cp_n

	# Calculate the receive energy, R
	R = np.zeros(d_range)
	for d in range(d_range):
		R[d] = 0
		for m in range(L):
			R[d] += np.power(np.abs(baseband_samples[d + m + L]), 2)
	# Normalize energy
	R *= SC_Scale
	

	# Normalize P1 
	P1_Norm = np.zeros(d_range)
	for d in range(d_range):
		x = np.power(R[d],2)
		if R[d] > 0.2:
			P1_Norm[d] = np.power(np.abs(P1_MA[d]), 2) / x
			
	P1_Derivative = np.zeros(d_range - 1)
	for d in range(d_range-1):
		P1_Derivative[d] = (P1_MA[d+1] - P1_MA[d])
	
	Sync_List = []
	Sync_Arm = 0
	Sync_Arm_Timer = 0
	Sync_Inhibit_Period = (2 * L) + cp_n
	Sync_Inhibit_Timer = Sync_Inhibit_Period
	Sync_Inhibit_Record = np.zeros(d_range)
	Sync_Arm_Record = np.zeros(d_range)
	for d in range(1,d_range-1):
		Sync_Inhibit_Timer += 1
		if Sync_Inhibit_Timer > Sync_Inhibit_Period:
			if P1_Norm[d] > 0.5:
				Sync_Arm = 1
				Sync_Arm_Timer = 0
			if Sync_Arm > 0:
				if (P1_Derivative[d] <= 0) & (P1_Derivative[d-1] > 0):
					Sync_List.append(d)
					Sync_Arm = 0
					Sync_Inhibit_Timer = 0
			Sync_Arm_Timer += 1
			if Sync_Arm_Timer >= 2 * L:
				Sync_Arm_Timer = 0
				Sync_Arm = 0
		Sync_Arm_Record[d] = Sync_Arm
		Sync_Inhibit_Record[d] = Sync_Inhibit_Timer - Sync_Inhibit_Period
			
	print('Coarse timing trigger samples:')
	print(Sync_List)


	fig, ax = plt.subplots(2,3, figsize=(12,8))
	fig.tight_layout()
	plt.subplot(234)
	plt.plot(baseband_samples.real, baseband_samples.imag, '.', ms=2)
	plt.title('Baseband Samples')
	plt.xlabel('in-phase')
	plt.ylabel('quadrature')
	plt.xlim(-0.5,0.5)
	plt.ylim(-0.5,0.5)
	plt.grid(True)
	plt.subplot(235)
	plt.plot(baseband_psd[0], baseband_psd[1], '.', ms=2)
	plt.xlim(-5000, 5000)
	plt.ylim(-60,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Baseband Spectrum")
	plt.grid(True)
	plt.subplot(231)
	plt.plot(audio_samples, linewidth=1)
	plt.title("Audio Samples")
	plt.ylim(-1.5,1.5)
	plt.subplot(232)
	plt.plot(audio_psd[0], audio_psd[1], '.', ms=2)
	plt.xlim(0, 4000)
	plt.ylim(-100,10)
	plt.ylabel("dBFS")
	plt.xlabel("Frequency, Hz")
	plt.title("Audio spectrum")
	plt.grid(True)

	plt.subplot(236)
	plt.title('Derivative of Moving Average of P1')
	plt.plot(P1_Derivative)
	plt.grid('true')

	plt.subplot(233)
	plt.plot(P1.real)
	plt.plot(P1_MA.real)
	plt.plot(R)
	plt.plot(P1_Norm)
	plt.plot(P1_Derivative)
	plt.plot(Sync_Arm_Record)
	plt.legend(['P','P Moving Avg','R Energy','M Final Metric', 'Derivative', 'Sync Arm'])
	plt.show()


	fft_freq = np.fft.fftfreq(fft_n, 1/audio_sample_rate)
	fft_freq = fft_freq[:fft_n//2]


	# Start sample for FFT should be in the center of the cyclic prefix
	
	sg = [[0,1],[0,2],[0,3],[0,4],[1,1],[1,2],[1,3],[1,4]]


	Error_Mags = np.zeros(bin_n)
	Error_Angles = np.zeros(bin_n)
	Avg_SNR_Lin = 0

	for SC_Peak_Sample in Sync_List:
		# Calculate reference baseband from known SC Preamble data
		Ref_BB = GenSCWidePreBB(fft_n, cp_n, sc_bin_0, sc_bin_max, bin_0, bin_max, 0)
		# Calculate reference error this will be zeros)
		Eq_BB = CalcEq(Ref_BB, Ref_BB)
		fig,ax = plt.subplots(2,5, figsize=(15,7), layout='constrained')
		plt.suptitle(f'Sample start: {SC_Peak_Sample}, pilot eq in green')
		#fig.tight_layout()

		SC_Offset = -cp_n // 2
		Start_i = SC_Peak_Sample + SC_Offset

		Sym_BB = []
		Sym_BB_Eq = []
		Sym_BB_Eq_x = []
		for sym_i in range(8):
			try:
				Sym_BB.append(np.fft.fft(baseband_samples[Start_i:Start_i + fft_n])*bin_n/fft_n)
			except:
				Sym_BB.append(np.zeros(fft_n, dtype='complex'))
			Start_i += (fft_n + cp_n)
		
			ax[sg[sym_i][0],sg[sym_i][1]].set_title(f'Symbol {sym_i}')
			ax[sg[sym_i][0],sg[sym_i][1]].set_xlim(-1.5,1.5)
			ax[sg[sym_i][0],sg[sym_i][1]].set_ylim(-1.5,1.5)
			ax[sg[sym_i][0],sg[sym_i][1]].grid(True)
		
			# Plot the unequalized subcarrier I/Q in grey
			#ax[sg[sym_i][0],sg[sym_i][1]].scatter(Sym_BB[sym_i][bin_0: bin_max+1].real,Sym_BB[sym_i][bin_0: bin_max+1].imag, color='grey', s=1)

			# Plot the unequalized pilots
			#if sym_i > 0: # no pilots in preamble
			#	for p in pilots:
			#		ax[sg[sym_i][0],sg[sym_i][1]].plot([0,Sym_BB[sym_i][p[0]].real],[0,Sym_BB[sym_i][p[0]].imag], color='grey', linewidth=1)

		# Collect equalization data from the Schmidle Cox preamble:
		Eq_BB, SNR_lin = CalcEq(Sym_BB[0],Ref_BB)
		Avg_SNR_Lin += SNR_lin
		SNR_dB = 10*np.log10(SNR_lin)

		for sym_i in range(8):
			
			# Refine the equalizer time offset based on the pilots of the previous symbol:

			Sym_BB_Eq.append(Sym_BB[sym_i] * Eq_BB)
			
			if sym_i > 1:
				p_eq = UpdateEqPilots(Sym_BB_Eq[sym_i], Eq_BB, pilots)
				Sym_BB_Eq_x.append(Sym_BB_Eq[sym_i] * p_eq)
			else:
				Sym_BB_Eq_x.append(Sym_BB_Eq[sym_i])
			
		
			# Plot the equalized subcarrier I/Q
			ax[sg[sym_i][0],sg[sym_i][1]].scatter(Sym_BB_Eq[sym_i][bin_0: bin_max+1].real,Sym_BB_Eq[sym_i][bin_0: bin_max+1].imag, color='red', s=2)
			#ax[sg[sym_i][0],sg[sym_i][1]].scatter(Sym_BB_Eq_x[sym_i][bin_0: bin_max+1].real,Sym_BB_Eq_x[sym_i][bin_0: bin_max+1].imag, color='green', s=2)
			
			# Plot the equalized pilots
			for p in pilots:
				if sym_i > 0: # no pilots in preamble
					ax[sg[sym_i][0],sg[sym_i][1]].plot([0,Sym_BB_Eq_x[sym_i][p[0]].real],[0,Sym_BB_Eq_x[sym_i][p[0]].imag], color='blue', linewidth=1)


		

		ax[0,0].set_title(f'Channel Magnitude\nPreamble SNR: {SNR_dB:.1f} dB')
		ax[0,0].scatter(fft_freq[bin_0: bin_max+1],np.abs(Sym_BB[0][bin_0: bin_max+1]), s=2, color='grey')
		# plot measured EQ points in blue, interpolated in red
		if (bin_0 % 2)==0: # bin_0 is even, first color blue
			first_color = 'blue'
			second_color = 'red'
			legend = ['preamble', 'meas eq', 'interp eq']
		else:
			first_color = 'red'
			second_color = 'blue'
			legend = ['preamble', 'interp eq', 'meas eq']
			
		ax[0,0].scatter(fft_freq[bin_0: bin_max+1:2],np.abs(Eq_BB[bin_0: bin_max+1:2]), s=2, color=first_color)
		ax[0,0].scatter(fft_freq[bin_0+1: bin_max+1:2],np.abs(Eq_BB[bin_0+1: bin_max+1:2]), s=2, color=second_color)
		ax[0,0].legend(legend)
		ax[0,0].set_ylim(-0.5,3.5)
		ax[0,0].grid(True)
		ax[1,0] = fig.add_subplot(2,5,6, projection='polar')
		ax[1,0].set_title('Channel Phase')
		ax[1,0].scatter(np.angle(Eq_BB[bin_0:bin_max+1:2]),fft_freq[bin_0:bin_max+1:2], s=2, color=first_color)
		ax[1,0].scatter(np.angle(Eq_BB[bin_0+1:bin_max+1:2]),fft_freq[bin_0+1:bin_max+1:2], s=2, color=second_color)
		ax[1,0].legend(legend[1:])
		ax[1,0].grid(True)

		plt.show()

		# Accumulate error data
		# create a list of pilot indices
		p_i = []
		for p in pilots:
			p_i.append(p[0])

		for sym_i in range(1,4,1):
			i = 0
			for bin_i in range(bin_0,bin_max+1,1):
				Error_Mags[i] += np.abs((np.abs(Sym_BB_Eq_x[sym_i][bin_i]) - 1))
				if bin_i in p_i:
					angle_tgt = 90
				else:
					angle_tgt = 45
				ea = np.abs(np.angle(Sym_BB_Eq_x[sym_i][bin_i], deg=True) - angle_tgt)
				while(ea > 45):
					ea -= 90
				Error_Angles[i] += abs(ea)
				i += 1
				
	# Plot all the data symbols on one I/Q chart:
	plt.figure()
	for sym_i in range(1,8):
		plt.scatter(Sym_BB_Eq_x[sym_i][bin_0: bin_max+1].real,Sym_BB_Eq_x[sym_i][bin_0: bin_max+1].imag,s=1, color='blue')
	plt.show()

	Avg_SNR_Lin /= len(Sync_List)
	Avg_SNR_dB = 10*np.log10(SNR_lin)

	Error_Freqs = fft_freq[bin_0:bin_max+1]
	Error_Sym_n = 3 * len(Sync_List)
	fig,ax = plt.subplots(1,2, layout='constrained')
	plt.suptitle(f'Error Analysis over {Error_Sym_n} Symbols\nMeasured SNR: {Avg_SNR_dB:.1f} dB')
	ax[0].set_title(f'Magnitude Error\nAvg {np.average(100*Error_Mags/Error_Sym_n):.1f}%')
	ax[0].scatter(Error_Freqs,100*Error_Mags/Error_Sym_n, s=6)
	ax[0].set_ylabel('Absolute Magnitude Error, %')
	ax[0].set_xlabel('Frequency, Hz')
	ax[0].set_ylim(0,50)
	ax[0].grid(True)
	ax[1].set_title(f'Angle Error\nAvg {np.average(Error_Angles/Error_Sym_n):.3f} deg')
	ax[1].scatter(Error_Freqs,Error_Angles/Error_Sym_n, s=6)
	ax[1].set_ylabel('Absolute Angle Error, Deg')
	ax[1].set_ylim(0,45)
	ax[1].set_xlabel('Frequency, Hz')
	ax[1].grid(True)
	plt.show()




if __name__ == "__main__":
	main()