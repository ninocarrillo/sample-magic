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

def SmoothSymbol(symbol, freq):
	paired = zip(freq,symbol)
	sorted_paired = sorted(paired)
	sorted_freq,sorted_symbol = zip(*sorted_paired)
	smooth_symbol = savgol_filter(np.abs(sorted_symbol),window_length=17, polyorder = 3) * np.exp(np.multiply(1j,np.angle(sorted_symbol)))
	final_symbol = np.zeros(len(smooth_symbol), dtype='complex')
	for i in range(len(smooth_symbol)):
		final_symbol[i] = smooth_symbol[np.where(sorted_freq == freq[i])]

	return final_symbol

def SelectSubcarriers(symbol):
	sel_symbol = symbol[48:81]
	sel_symbol = np.concatenate([sel_symbol,symbol[17:48]])
	return sel_symbol

def PadBasebandZeros(symbol, rate):
	padded_symbol = np.zeros(len(symbol) * rate, dtype='complex')
	for i in range(len(symbol)):
		padded_symbol[i] = symbol[i]
	return padded_symbol

def CalcSubcarrierError(symbol):
	FRS_Phase = [135,-90,45,135,-45,45,-45,0,-45,135,-45,45,135,-135,45,-45,135,-45,-45,135,-135,0,-45,-45,-135,-135,-45,-45,-135,45,-135,45,135,135,-135,135,-135,45,-45,-45,135,-45,45,0,-135,-135,135,135,-45,45,45,-45,-135,-135,45,-45,45,-180,-45,-135,-135,45,-135,-135]
	FRS_Symbol = np.exp(np.multiply(1j*np.pi/180,FRS_Phase))
	Error_Vector = FRS_Symbol.conj() * symbol
	Error_Vector = Error_Vector * (1/np.power(np.abs(Error_Vector),2))
	# Remove data for empty subcarriers
	Error_Vector[0] = Error_Vector[2]
	Error_Vector[1] = Error_Vector[2]
	Error_Vector[32] = Error_Vector[33]
	return Error_Vector
	
def CalcPilotError(symbol, bb_fs, oversample):
	p0 = 7 # 0 Phase
	p1 = 21 # 0 Phase
	p2 = 43 # 0 Phase
	p3 = 57 # 180 Phase
	
	# Calculate phase error of each pilot in fine ranging symbol
	p0_err = np.angle(symbol[p0], deg=True)
	p1_err = np.angle(symbol[p1], deg=True)
	p2_err = -np.angle(symbol[p2], deg=True)
	p3_err = -180-np.angle(symbol[p3], deg=True)
	while p3_err <= -180:
		p3_err += 360
	while p3_err > 180:
		p3_err -= 360
	while p2_err <= -180:
		p2_err += 360
	while p2_err > 180:
		p2_err -= 360
	while p1_err <= -180:
		p1_err += 360
	while p1_err > 180:
		p1err -= 360
	while p0_err <= -180:
		p0_err += 360
	while p0_err > 180:
		p0_err -= 360
		
	# Normalize phase error to subcarrier index position
	p0_err_norm = p0_err / (p0+1)
	p1_err_norm = p1_err / (p1+1)
	p2_err_norm = p2_err / (64-p2)
	p3_err_norm = p3_err / (64-p3)

	fine_range_exact = np.average([p0_err_norm, p1_err_norm, p2_err_norm, p3_err_norm]) # degrees per subcarrier

	# convert the fine range estimate to time units
	print(f'Baseband Sample Rate: {bb_fs} Hz')
	bin_spacing = bb_fs / (len(symbol) * oversample)
	print(f'Bin spacing: {bin_spacing} Hz')
	time_offset = fine_range_exact / (bin_spacing * 360)
	pilot_sample_offset = time_offset * bb_fs
	print(f'Sample time offset: {time_offset * 1e3:.2f} ms, {pilot_sample_offset:.2f} samples')


	print(f'Pilot 0 Error: {p0_err:.2f} deg, {p0_err_norm:.2f} deg/sub')
	print(f'Pilot 1 Error: {p1_err:.2f} deg, {p1_err_norm:.2f} deg/sub')
	print(f'Pilot 2 Error: {p2_err:.2f} deg, {p2_err_norm:.2f} deg/sub')
	print(f'Pilot 3 Error: {p3_err:.2f} deg, {p3_err_norm:.2f} deg/sub')
	print(f'Average pilot error: {fine_range_exact:.2f} deg/sub')
	
	fft_n = len(symbol)
	Phase_Error = np.zeros(fft_n)
	Feedback_Gain = 0.75
	for i in range(fft_n):
		if i < fft_n/2:
			Phase_Error[i] =  (i + 1) * time_offset * (bin_spacing * 360) * Feedback_Gain
		else:
			Phase_Error[i] = -(fft_n - i) * time_offset * (bin_spacing * 360) * Feedback_Gain
	
	Error_Vector = np.exp(np.multiply(1j*np.pi/180,Phase_Error))
			
	
	
	return pilot_sample_offset, Error_Vector

def AvgSubcarriers(symbol, carrier_list):
	y = 0j
	for i in carrier_list:
		y += symbol[i]
	return y / len(carrier_list)
	
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

	try:
		audio_sample_rate, audio_samples  = readwav(sys.argv[1])
	except:
		print('Unable to open audio file.')
		sys.exit(3)

	# Normalize audio amplitude
	if audio_samples.dtype == np.int16:
		audio_samples = audio_samples / 32767

	audio_sample_count = len(audio_samples)
	
	sym_rate = audio_sample_rate / (fft_n + cp_n)
	bin_width = audio_sample_rate / fft_n

	print(f'Audio Sample Rate: {audio_sample_rate}')
	print(f'FFT N: {fft_n}')
	print(f'Bin Width: {bin_width}')
	print(f'Symbol Rate: {sym_rate:.2f}')
	print(f'Cyclic Prefix time: {1000*cp_n/audio_sample_rate:.2f} mS, {100*cp_n / (cp_n + fft_n):.1f}%')


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
		if R[d] > 0.01:
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



	# Start sample for FFT should be in the center of the cyclic prefix
	for fudge_offset in range(cp_n, cp_n+2):

		fudge = fudge_offset

		for SC_Peak_Sample in Sync_List:
			SC_Offset = (2 * L) + fudge
			Start_i = SC_Peak_Sample + SC_Offset
			Start_i -= (fft_n + cp_n)

			# Collect and process the fine ranging symbol
			Symbol_Baseband = baseband_samples[Start_i:Start_i + fft_n]

			Symbol_Output = np.fft.fft(Symbol_Baseband, fft_n) * (bin_n / fft_n)
			
			
			fig,ax = plt.subplots(2,3, figsize=(12,8))
			plt.suptitle(f'Fudge: {fudge}')
			fig.tight_layout()
			plt.subplot(231)
			plt.scatter(freq[bin_0:bin_max],np.abs(Symbol_Output[bin_0:bin_max]), s=2)
			plt.ylim(0,2)
			plt.subplot(234)
			plt.scatter(freq[bin_0:bin_max],np.angle(Symbol_Output[bin_0:bin_max]), s=2)
			plt.ylim(-np.pi,np.pi)
			plt.subplot(232)
			plt.scatter(Symbol_Output[bin_0:bin_max].real, Symbol_Output[bin_0:bin_max].imag, s=2)
			plt.xlim(-1.5,1.5)
			plt.ylim(-1.5,1.5)
			plt.grid(True)
			plt.show()



if __name__ == "__main__":
	main()