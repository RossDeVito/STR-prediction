import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
	motif_name = 'AC-AG-AT-CT-GT'

	# Load data
	data_dir = os.path.join('..', '..', 'data', 'heterozygosity')
	data_fname = 'sample_data_{}_V2_repeat_var.json'.format(motif_name)

	with open(os.path.join(data_dir, data_fname)) as fp:    
		samples = json.load(fp)

	data_df = pd.DataFrame(samples)

	# Plot distributions
	sns.displot(data_df.label, kde=True)
	plt.title('Heterozygosity distribution')
	plt.tight_layout()
	plt.savefig('{}_heterozygosity_dist.png'.format(motif_name))
	plt.show()

	counts_df = data_df.groupby(['num_copies',  'binary_label']).count()
	sns.barplot(x='num_copies', y='str_seq', hue='binary_label', data=counts_df.reset_index())
	plt.gcf().set_size_inches(10, 6)
	plt.ylabel('number of samples')
	plt.setp(plt.gca().get_xticklabels()[1::2], visible=False)
	plt.title('Heterozygosity by Copy Number')
	plt.tight_layout()
	plt.savefig('{}_heterozygosity_by_len.png'.format(motif_name))
	plt.show()

	# STR len vs heterozygosity
	sns.lineplot(x='num_copies', y='label', data=data_df, ci='sd')
	plt.ylabel('heterozygosity')
	plt.title('Heterozygosity vs STR Length (std. dev. interval)')
	plt.tight_layout()
	plt.savefig('{}_heterozygosity_v_len.png'.format(motif_name))
	plt.show()

	# STR len vs heterozygosity heatmap
	sns.displot(x='num_copies', y='label', data=data_df[data_df.binary_label == 1])
	plt.ylabel('heterozygosity')
	plt.title('Heterozygosity vs STR Length Heatmap\n(heterozygous samples only)')
	plt.tight_layout()
	plt.savefig('{}_heterozygosity_v_len_heatmap.png'.format(motif_name))
	plt.show()

	# Now with filtering by len
	data_df[data_df.num_copies <= 11.5].binary_label.value_counts()
	data_df[(7 <= data_df.num_copies) & (data_df.num_copies <= 15)].binary_label.value_counts()
