from scipy.io import wavfile
import os
import numpy as np
import argparse
from tqdm import tqdm

# Utility functions

def windows(signal, window_size, step_size):
    if type(window_size) is not int:
        raise AttributeError("Window size must be an integer.")
    if type(step_size) is not int:
        raise AttributeError("Step size must be an integer.")
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]

def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))

def rising_edges(binary_signal):
    previous_value = 0
    index = 0
    for x in binary_signal:
        if x and not previous_value:
            yield index
        previous_value = x
        index += 1


'''
Split a WAV file at silence.

Prams:
'input_file', type=str, help='The WAV file to split.'
'output_dir', type=str, help='The output folder. Defaults to the current folder.'
'window_duration', type=float, help='The minimum length of silence at which a split may occur [seconds].'
'silence_threshold', type=float, default=1e-6, help='The energy level (between 0.0 and 1.0) below which the signal is regarded as silent. Defaults to 1e-6 == 0.0001%.')
step_duration', type=float, default=None, help='The amount of time to step forward in the input file after calculating energy. Smaller value = slower, but more accurate silence detection. Larger value = faster, but might miss some split opportunities. Defaults to (min-silence-length / 10.).
'dry_run', help='Don\'t actually write any output files.'
'''
def split(input_file, output_dir, window_duration, dry_run, silence_threshold=1e-6, step_duration=None):
    input_filename = input_file
    if step_duration is None:
        step_duration = window_duration / 10.
    output_filename_prefix = os.path.splitext(os.path.basename(input_filename))[0]
    
    
    print("Splitting {} where energy is below {}% for longer than {}s.".format(
        input_filename,
        silence_threshold * 100.,
        window_duration
    ))
    
    # Read and split the file
    
    sample_rate, samples = input_data=wavfile.read(filename=input_filename, mmap=True)
    
    max_amplitude = np.iinfo(samples.dtype).max
    max_energy = energy([max_amplitude])
    
    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)
    
    signal_windows = windows(
        signal=samples,
        window_size=window_size,
        step_size=step_size
    )
    
    window_energy = (energy(w) / max_energy for w in tqdm(
        signal_windows,
        total=int(len(samples) / float(step_size))
    ))
    
    window_silence = (e > silence_threshold for e in window_energy)
    
    cut_times = (r * step_duration for r in rising_edges(window_silence))
    
    # This is the step that takes long, since we force the generators to run.
    print("Finding silences...")
    cut_samples = [int(t * sample_rate) for t in cut_times]
    cut_samples.append(-1)
    
    cut_ranges = [(i, cut_samples[i], cut_samples[i+1]) for i in range(len(cut_samples) - 1)]
    
    for i, start, stop in tqdm(cut_ranges):
        output_file_path = "{}_{:03d}.wav".format(
            os.path.join(output_dir, output_filename_prefix),
            i
        )
        if not dry_run:
            print("Writing file {}".format(output_file_path))
            wavfile.write(
                filename=output_file_path,
                rate=sample_rate,
                data=samples[start:stop]
            )
        else:
            print("Not writing file {}".format(output_file_path))
            
if __name__ == "__main__":
    input_file = './data/62402__bsumusictech__russians-exiting-bathroom.wav'
    output_dir = './splited_wav/'
    window_duration = 10
    dry_run  = False
    
    split(input_file, output_dir, window_duration, dry_run)
    