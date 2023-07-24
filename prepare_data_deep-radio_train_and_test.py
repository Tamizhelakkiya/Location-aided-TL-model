import argparse, time, random, string, os, sys
import numpy as np
import scipy.signal as signal
from rtlsdr import RtlSdr

def build_parser():
    parser = argparse.ArgumentParser(description='Prepare the data')
    parser.add_argument('-decimation_rate', dest = 'decimation_rate', type = int, 
         default = 12, help = 'Decimation rate of the signal')
    parser.add_argument('-sampling_rate', dest = 'sampling_rate', type = int, 
         default = 2400000, help = 'Sampling rate of the signal')
    parser.add_argument('-sdr', dest = 'sdr', type = int, 
         default = 1, help = 'Read samples from file (0) or device (1)')
    return parser



def prepare_args():
    # hack, http://stackoverflow.com/questions/9025204/
    for i, arg in enumerate(sys.argv):
        if (arg[0] == '-') and arg[1].isdigit():
            sys.argv[i] = ' ' + arg
    parser = build_parser()
    args = parser.parse_args()
    return args
    
    
def read_samples_sdr(freq):
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.err_ppm = 56   # change it to yours
    sdr.gain = "auto"

    f_offset = 250000 # shifted tune to avoid DC
    sdr.center_freq = freq - f_offset
    time.sleep(0.06)
    iq_samples = sdr.read_samples(1221376)
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples



def read_samples(freq):
    f_offset = 250000  # shifted tune to avoid DC
    #center_freq = freq - f_offset
    #time.sleep(0.06)
    samp = np.fromfile(str(freq)+'_training_samples.dat',np.uint8)+np.int8(-127) #sdr.read_samples(1221376)
    x1 = samp[::2]/128
    x2 = samp[1::2]/128
    iq_samples = x1+x2*1j
    iq_samples = iq_samples[0:600000]
    fc1 = np.exp(-1.0j * 2.0 * np.pi * f_offset / sample_rate * np.arange(len(iq_samples)))  # shift down 250kHz
    iq_samples = iq_samples * fc1
    return iq_samples

def randomword(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def collect_samples(freq, classname, decimation_rate):
    os.makedirs("training_data/" + classname, exist_ok=True)
    os.makedirs("testing_data/" + classname, exist_ok=True)
    for i in range(0, 500):
        if args.sdr == 1:
            iq_samples = read_samples_sdr(freq)
        elif args.sdr == 0:
            iq_samples = read_samples(freq)
        iq_samples = signal.decimate(iq_samples, decimation_rate, zero_phase=True)
        if (i < 375):  # 75% train, 25% test
            filename = "training_data/" + classname + "/samples-" + randomword(16) + ".npy"
        else:
            filename = "testing_data/" + classname + "/samples-" + randomword(16) + ".npy"
        np.save(filename, iq_samples)
        if not (i % 5): print(i / 5, "%", classname)



args = prepare_args()  
sample_rate = args.sampling_rate
decimation_rate = args.decimation_rate

#collect_samples(93500000, "wfm",decimation_rate)
##collect_samples(106400000, "wfm",decimation_rate)
#collect_samples(939600000, "gsm",decimation_rate)
#collect_samples(940200000, "gsm",decimation_rate)
#collect_samples(940600000, "gsm",decimation_rate)
#collect_samples(880000000, "lte",decimation_rate)
#collect_samples(884000000, "lte",decimation_rate)
#collect_samples(112000000,"other",decimation_rate)
#collect_samples(450000000,"other",decimation_rate)

for i in range (1,8):
    x=input('Enter class name:')
    collect_samples(810000000,x,decimation_rate)
    
    


