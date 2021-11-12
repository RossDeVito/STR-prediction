import json
import os

import numpy as np
import pandas as pd
import scipy.stats as stats

from tqdm import tqdm


if __name__ == '__main__':
	plot_label_dist = True

	# Load target STRs to be labeled
	unlabeled_samp_dir = os.path.join('..', 'data', 'HipSTR-references')
	unlabeled_samp_fname = 'unlabeled_samples_GRCh38_500_per_side.json'
	unlabeled_samp_path = os.path.join(unlabeled_samp_dir, unlabeled_samp_fname)

	data_dir = os.path.join('..', 'data', 'repeat_num')

	with open(unlabeled_samp_path) as fp:    
		samples = json.load(fp)

	new_samples = []
	all_labels = []

	# Filter out super long repeats
	max_repeat_len = 40

	# Label each STR sample with heterozygosity
	for samp in tqdm(samples):
		num_copies = samp['num_copies']
		if num_copies <= max_repeat_len:
			samp['label'] = num_copies
			new_samples.append(samp)
			all_labels.append(num_copies)

	print(len(new_samples))

	# Save labeled samples to main data dict
	this_sample_set_fname = 'labeled_samples_repeat_num.json'
	samples_save_path = os.path.join(data_dir, this_sample_set_fname)

	with open(samples_save_path, 'w') as fp:
		json.dump(new_samples, fp, indent=4)

	if plot_label_dist:
		import matplotlib.pyplot as plt
		import seaborn as sns

		sns.displot(np.array(all_labels))
		plt.show()