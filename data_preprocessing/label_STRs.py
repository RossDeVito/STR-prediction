import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
	# Load target STRs to be labeled
	unlabeled_samp_dir = os.path.join('..', 'data', 'mecp2_binding')
	unlabeled_samp_fname = 'unlabeled_samples_GRCh38_500_per_side.json'
	unlabeled_samp_path = os.path.join(unlabeled_samp_dir, unlabeled_samp_fname)

	with open(unlabeled_samp_path) as fp:    
		samples = json.load(fp)

	# Load ChIP-seq data
	chipseq_dir = os.path.join('..', 'data', 'ChIP-seq')
	peaks_bed_fname = 'GSM3579716_GSM3579716_ChIP5_IP_WT_6.bed'

	peaks_df = pd.read_csv(
		os.path.join(chipseq_dir, peaks_bed_fname), 
		sep='\t', 
		names=['chr', 'start', 'stop', 'peak_name', 
				'score', 'strand', 'signal_value',
				'p_value', 'q_value', 'peak_offset'],
		skiprows=1
	)
	print("total peaks:\t{}".format(len(peaks_df)))

	# Get peak start and end locs by chromosome. Will be used to find 
	# overlap with STR regions.
	included_chroms = list(set(s['str_seq_name'].split(':')[0] for s in samples))

	start_pos = dict()
	end_pos = dict()

	for chrom in included_chroms:
		chrom_peaks = peaks_df[peaks_df.chr == chrom]
		start_pos[chrom] = chrom_peaks.start.values
		end_pos[chrom] = chrom_peaks.stop.values

	# Label each STR sample with 1 if bound (overlaps peak range) 
	# or 0 otherwise
	bound_counts = {True: 0, False: 0}

	for i in tqdm(range(len(samples))):
		samp = samples[i]
		chrom, pos = samp['str_seq_name'].split(':')
		start, end = (int(val) for val in pos.split('-'))
		
		samp_start_leq_peak_end = start <= end_pos[chrom]
		samp_end_geq_peak_start = end >= start_pos[chrom]

		is_bound = np.any(samp_start_leq_peak_end & samp_end_geq_peak_start)
		bound_counts[is_bound] += 1

		# Update list of dicts with label
		samples[i]['label'] = int(is_bound)

	print(bound_counts)

	# Save unlabeled samples to main data dict
	this_sample_set_fname = 'labeled_samples_GRCh38_500_per_side.json'
	samples_save_path = os.path.join('..', 'data', 'mecp2_binding', 
										this_sample_set_fname)

	with open(samples_save_path, 'w') as fp:
		json.dump(samples, fp, indent=4)