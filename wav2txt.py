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


def CalcSubcarrierError(symbol):
	FRS_Phase = [135,-90,45,135,-45,45,-45,0,-45,135,-45,45,135,-135,45,-45,135,-45,-45,135,-135,0,-45,-45,-135,-135,-45,-45,-135,45,-135,45,135,135,-135,135,-135,45,-45,-45,135,-45,45,0,-135,-135,135,135,-45,45,45,-45,-135,-135,45,-45,45,-180,-45,-135,-135,45,-135,-135]
	FRS_Symbol = np.exp(np.multiply(1j*np.pi/180,FRS_Phase))
	Error_Vector = FRS_Symbol.conj() * symbol
	Error_Vector = Error_Vector * (1/np.power(np.abs(Error_Vector),2))
	Phase_Error = np.angle(Error_Vector) * 180 / np.pi
	Mag_Error = np.abs(symbol) - np.abs(FRS_Symbol)
	# Remove data for empty subcarriers
	Error_Vector[0] = 0
	Error_Vector[1] = 0
	Error_Vector[32] = 0
	return Error_Vector, Phase_Error, Mag_Error
	
def CalcPilotError(symbol, bb_fs, oversample):
	p0 = 7 # 0 Phase
	p1 = 21 # 0 Phase
	p2 = 43 # 0 Phase
	p3 = 57 # 180 Phase
	lo_phase_correction = (symbol[p0].conj()-symbol[p3].conj()) / 2
	
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
	return pilot_sample_offset

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
	carrier_phase = 0
	for i in range(audio_sample_count):
		time_var = 2 * np.pi * i * (-carrier_freq) / audio_sample_rate
		baseband_samples[i] = audio_samples[i] * np.exp((time_var + carrier_phase) * 1j)

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
	Oversample = int(baseband_sample_rate / 2000)
	L = int(Oversample * FFT_N / 2)
	
	# Create empty list to place metric P values
	d_range = len(baseband_samples) - (FFT_N * Oversample)
	P1 = np.zeros(d_range, dtype='complex')
	for d in range(d_range):
		P1[d] = 0
		for m in range(L):
			P1[d] += baseband_samples[d + m].conj()*baseband_samples[d + m + L]
	
	# Normalize the P1 metric based on oversample rate
	P1 = P1 / Oversample
			
	# Discard imaginary part of P1
	P1 = P1.real

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
			
	P1_Derivative = np.zeros(d_range - 1)
	for d in range(d_range-1):
		P1_Derivative[d] = P1_MA[d+1] - P1_MA[d]
	
	Sync_List = []
	Sync_Arm = 0
	Sync_Arm_Timer = 0
	Sync_Inhibit_Period = (2 * L) + (CP_N * Oversample)
	Sync_Inhibit_Timer = Sync_Inhibit_Period
	for d in range(1,d_range-1):
		Sync_Inhibit_Timer += 1
		if Sync_Inhibit_Timer > Sync_Inhibit_Period:
			if P1_Norm[d] > 0.5:
				Sync_Arm = 1
			if Sync_Arm > 0:
				if (P1_Derivative[d] < 0) & (P1_Derivative[d-1] > 0):
					Sync_List.append(d)
					Sync_Arm = 0
					Sync_Inhibit_Timer = 0
			Sync_Arm_Timer += 1
			if Sync_Arm_Timer >= 2 * L:
				Sync_Arm_Timer = 0
				Sync_Arm = 0
			
	print('Coarse timing trigger samples:')
	print(Sync_List)

	plt.figure()
	plt.title('Derivative of Moving Average of P1')
	plt.plot(P1_Derivative)
	plt.grid('true')
	plt.show()

	plt.figure()
	plt.plot(np.abs(baseband_samples))
	plt.plot(P1.real)
	plt.plot(P1_MA.real)
	plt.plot(R)
	plt.plot(P1_Norm)
	plt.plot(P1_Derivative)
	plt.legend(['Baseband','P','P Moving Avg','R Energy','M Final Metric', 'Derivative'])
	plt.show()


	# Decimate from selected sample
	
	# Location of pilot subcarriers
	pilot_index = [7,21,43,57]

	CP_Length = 8 * Oversample
	for SC_Peak_Sample in Sync_List:
		SC_Offset = (2 * L) + CP_Length + int(1.5 * Oversample)
		Start_i = SC_Peak_Sample + SC_Offset

		# Collect and process the fine ranging symbol
		Symbol_Baseband = baseband_samples[Start_i:Start_i + (Oversample * FFT_N):Oversample]
		Symbol_Output = np.fft.fft(Symbol_Baseband, FFT_N)

		FRS_Error_Symbol, FRS_Phase_Error, FRS_Mag_Error = CalcSubcarrierError(Symbol_Output)

		Corrected_Symbol = Symbol_Output * FRS_Error_Symbol.conj()

		Corrected_Error_Symbol, Corrected_Phase_Error, Corrected_Mag_Error = CalcSubcarrierError(Corrected_Symbol)


		# plot Fine Ranging Symbol phase and magnitude erros
		fig,ax = plt.subplots(2,1)
		plt.suptitle("Fine Ranging Symbol Analysis")
		plt.subplot(211)
		plt.title("Phase Angle Error, Degrees")
		plt.grid(True)
		plt.xticks([7,21,FFT_N/2,43,57])
		plt.yticks([-180,-134,-90,-45,0,45,90,135,180])
		plt.ylim(-200,200)
		plt.scatter(np.linspace(0,FFT_N,num=FFT_N,endpoint=False),FRS_Phase_Error,s=2)
		plt.scatter(np.linspace(0,FFT_N,num=FFT_N,endpoint=False),Corrected_Phase_Error,s=2)
		plt.legend(['Unequalized', 'Equalized'])
		plt.subplot(212)
		plt.title("Magnitude Error")
		plt.grid(True)
		plt.xticks([7,21,FFT_N/2,43,57])
		plt.ylim(-1.5,1.5)

		plt.scatter(np.linspace(0,FFT_N,num=FFT_N,endpoint=False),FRS_Mag_Error,s=2)
		plt.scatter(np.linspace(0,FFT_N,num=FFT_N,endpoint=False),Corrected_Mag_Error,s=2)
		plt.legend(['Unequalized', 'Equalized'])
		plt.show()

		Eq_Symbol_Output = Symbol_Output * FRS_Error_Symbol.conj()
		CalcPilotError(Eq_Symbol_Output, baseband_sample_rate, Oversample)

		Symbol2_Baseband = baseband_samples[Start_i+(Oversample * (FFT_N+8)):(Oversample*(FFT_N+8))+Start_i + (Oversample * FFT_N):Oversample]
		Symbol2_Output = np.fft.fft(Symbol2_Baseband, FFT_N)
		Eq_Symbol2_Output = Symbol2_Output * FRS_Error_Symbol.conj()
		s2_adj = int(round(CalcPilotError(Eq_Symbol2_Output, baseband_sample_rate, Oversample)))
		Start_i -= s2_adj
		
		Symbol3_Baseband = baseband_samples[Start_i+(2*Oversample * (FFT_N+8)):(2*Oversample*(FFT_N+8))+Start_i + (Oversample * FFT_N):Oversample]
		Symbol3_Output = np.fft.fft(Symbol3_Baseband, FFT_N)
		Eq_Symbol3_Output = Symbol3_Output * FRS_Error_Symbol.conj()
		s3_adj = int(round(CalcPilotError(Eq_Symbol3_Output, baseband_sample_rate, Oversample)))
		Start_i -= s3_adj
		
		Symbol4_Baseband = baseband_samples[Start_i+(3*Oversample * (FFT_N+8)):(3*Oversample*(FFT_N+8))+Start_i + (Oversample * FFT_N):Oversample]
		Symbol4_Output = np.fft.fft(Symbol4_Baseband, FFT_N)
		Eq_Symbol4_Output = Symbol4_Output * FRS_Error_Symbol.conj()
		
		Subcarrier_List = []
		j = 0
		for i in range(FFT_N):
			if i != 0:
				if i != 1:
					if i != 32:
						Subcarrier_List.append(i)
						
		print(f'Subcarrier list is {len(Subcarrier_List)} elements long.')
			
		print(f'Fine Ranging Symbol Average Phase Error: {np.angle(AvgSubcarriers(FRS_Error_Symbol, Subcarrier_List), deg=True):.1f}')
		
		fig,ax = plt.subplots(2,4)
		plt.suptitle(f'Start Index {Start_i}, Oversample {Oversample}, LO Phase Error {np.angle(AvgSubcarriers(FRS_Error_Symbol, Subcarrier_List), deg=True):.1f} deg\nUnequalized in grey, equalized in red')
		fig.tight_layout()
		plt.subplot(241)
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.grid('true')
		plt.title(f'Fine Ranging Symbol\nPilots')
		for i in pilot_index:
			plt.plot([0,Symbol_Output[i].real],[0,Symbol_Output[i].imag], color='grey')
			plt.plot([0,Eq_Symbol_Output[i].real],[0,Eq_Symbol_Output[i].imag], color='red')
		plt.subplot(242)
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.grid('true')
		plt.title(f'Symbol 2Pilots')
		for i in pilot_index:
			plt.plot([0,Symbol2_Output[i].real],[0,Symbol2_Output[i].imag], color='grey')
			plt.plot([0,Eq_Symbol2_Output[i].real],[0,Eq_Symbol2_Output[i].imag], color='red')
		plt.subplot(243)
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.grid('true')
		plt.title(f'Symbol 3 Pilots\nAdj {s2_adj} samps')
		for i in pilot_index:
			plt.plot([0,Symbol3_Output[i].real],[0,Symbol3_Output[i].imag], color='grey')
			plt.plot([0,Eq_Symbol3_Output[i].real],[0,Eq_Symbol3_Output[i].imag], color='red')
		plt.subplot(244)
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.grid('true')
		plt.title(f'Symbol 4 Pilots\nAdj {s3_adj} samps')
		for i in pilot_index:
			plt.plot([0,Symbol4_Output[i].real],[0,Symbol4_Output[i].imag], color='grey')
			plt.plot([0,Eq_Symbol4_Output[i].real],[0,Eq_Symbol4_Output[i].imag], color='red')
		plt.subplot(245)
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.grid('true')
		plt.title(f'Fine Ranging Symbol\nData Subcarriers')
		plt.scatter(Symbol_Output.real, Symbol_Output.imag, s=3, color='grey')
		plt.scatter(Eq_Symbol_Output.real, Eq_Symbol_Output.imag, s=6, color='red')
		plt.subplot(246)
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.grid('true')
		plt.title(f'Symbol 2\nData Subcarriers')
		plt.scatter(Symbol2_Output.real, Symbol2_Output.imag, s=3, color='grey')
		plt.scatter(Eq_Symbol2_Output.real, Eq_Symbol2_Output.imag, s=6, color='red')
		plt.subplot(247)
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.grid('true')
		plt.title(f'Symbol 3\nData Subcarriers')
		plt.scatter(Symbol3_Output.real, Symbol3_Output.imag, s=3, color='grey')
		plt.scatter(Eq_Symbol3_Output.real, Eq_Symbol3_Output.imag, s=6, color='red')
		plt.subplot(248)
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.grid('true')
		plt.title(f'Symbol 4\nData Subcarriers')
		plt.scatter(Symbol4_Output.real, Symbol4_Output.imag, s=3, color='grey')
		plt.scatter(Eq_Symbol4_Output.real, Eq_Symbol4_Output.imag, s=6, color='red')
		plt.show()

if __name__ == "__main__":
	main()