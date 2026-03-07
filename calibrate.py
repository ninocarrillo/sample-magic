# Python3
# Nino Carrillo
# 4 Mar 26

import sys
from scipy.io.wavfile import read as readwav

def main():
	# check correct version of Python
	if sys.version_info < (3, 0):
		print("Python version should be 3.x, exiting")
		sys.exit(1)
	# check correct number of parameters were passed to command line
	if len(sys.argv) != 3:
		print("Incorrect arg count. Usage: python3 calibrate.py <input wav file> <carrier freq>")
		sys.exit(2)

	try:
		audio_sample_rate, audio_samples  = readwav(sys.argv[1])
	except:
		print('Unable to open audio file.')
		sys.exit(3)

	audio_sample_count = len(audio_samples)
	carrier_freq = float(sys.argv[2])
	crossing_count = 0
	state = 0
	for sample in audio_samples:
		if state == 1:
			if sample < 0:
				state = 0
		else:
			if sample >= 0:
				state = 1
				crossing_count+= 1
	
	frequency = crossing_count * audio_sample_rate / audio_sample_count
	
	frequency_error = (frequency - carrier_freq) / carrier_freq
	
	print(f'Measured frequency is {frequency:.6f} Hz.')
	print(f'Frequency error is {(frequency_error * 1e6):.1f} ppm')

				

if __name__ == "__main__":
	main()