import json
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
# import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm


if __name__ == '__main__':
	# Load target STRs to be labeled
	unlabeled_samp_dir = os.path.join('..', 'data', 'HipSTR-references')
	unlabeled_samp_fname = 'unlabeled_samples_GRCh38_500_per_side.json'
	unlabeled_samp_path = os.path.join(unlabeled_samp_dir, unlabeled_samp_fname)

	with open(unlabeled_samp_path) as fp:    
		samples = json.load(fp)

	# Load heterozygosity data from statSTR
	data_dir = os.path.join('..', 'data', 'heterozygosity')
	data_fname = 'CEU_filtered_strhet.tab'

	data_df = pd.read_csv(
		os.path.join(data_dir, data_fname), 
		sep='\t', 
		header=0
	)
	data_df = data_df.dropna()
	print("total labeled data points:\t{}".format(len(data_df)))

	# # Plot distributions
	# sns.displot(data_df.het, kde=True)
	# hets = data_df.het.values.copy()
	# hets[hets == 0.0] = .001
	# sns.displot(np.log(hets), kde=True)
	# plt.show()

	# Get peak start and end locs by chromosome. Will be used to find 
	# overlap with STR regions.
	included_chroms = list(set(s['str_seq_name'].split(':')[0] for s in samples))

	chrom_dfs = dict()

	for chrom in included_chroms:
		chrom_dfs[chrom] = data_df[data_df.chrom == chrom]

	new_samples = []
	n_new_samples = 0
	n_called = []

	# Label each STR sample with heterozygosity
	for samp in tqdm(samples):
		chrom, pos = samp['str_seq_name'].split(':')
		start, end = (int(val) for val in pos.split('-'))

		samp_leq_range_end = start <= chrom_dfs[chrom].end.values
		samp_geq_range_start = end >= chrom_dfs[chrom].start.values
		

		in_range = samp_leq_range_end & samp_geq_range_start
		assert in_range.sum() == int(in_range.any())

		if in_range.any():
			range_ind = np.argwhere(in_range == True)[0]
			samp['label'] = float(chrom_dfs[chrom].iloc[range_ind].het)
			assert not np.isnan(samp['label'])
			samp['num_called'] = int(chrom_dfs[chrom].iloc[range_ind].numcalled)
			n_called.append(samp['num_called'])
			new_samples.append(samp)

	print(len(new_samples))

	# Save labeled samples to main data dict
	this_sample_set_fname = 'labeled_samples_het.json'
	samples_save_path = os.path.join(data_dir, this_sample_set_fname)

	with open(samples_save_path, 'w') as fp:
		json.dump(new_samples, fp, indent=4)

	# sns.displot(np.array(n_called))
	# plt.show()