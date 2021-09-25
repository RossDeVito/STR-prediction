"""Finds target STR types in BED output from HipSTR and extracts
surrounding sequence from reference genome. Saves resuting samples
to a JSON file in main data dir.
"""
import os
import json

import pandas as pd
from tqdm import tqdm

from pyfaidx import Fasta


if __name__ == '__main__':
	# Load STR region BED
	str_region_bed_fname = 'GRCh38.hipstr_reference.bed.gz'
	str_region_bed_path = os.path.join('..', 'data', 'HipSTR-references', 
										str_region_bed_fname)
	str_regions = pd.read_csv(
		str_region_bed_path, 
		sep='\t', 
		names=['chr', 'start', 'stop', 'motif_len', 
				'num_copies', 'str_name', 'motif'],
		low_memory=False, # because chr field is mixed type
	)
	print("total STRs:\t{}".format(len(str_regions)))

	# Filter down to relevant motifs
	target_motifs = ['AC', 'CA', 'TG', 'GT']
	str_regions = str_regions[str_regions.motif.isin(target_motifs)]

	print("found target regions:")
	print(str_regions.motif.value_counts())

	# Load reference genome
	ref_fasta_path = '../data/human-references/GRCh38_full_analysis_set_plus_decoy_hla.fa'
	ref_genome = Fasta(ref_fasta_path)

	# Extract sequence around each relevant STR region
	n_per_side = 500	# number of additional allels on either side of STR seq

	samples = []

	for _, region in tqdm(str_regions.iterrows(), 
							desc='Getting sample regions',
							total=len(str_regions)):
		chr_str = 'chr{}'.format(region.chr)
		str_seq = ref_genome[chr_str][region.start-1 : region.stop]
		pre_seq = ref_genome[chr_str][region.start-1-n_per_side : region.start-1]
		post_seq = ref_genome[chr_str][region.stop : region.stop+n_per_side]
		full_seq = ref_genome[chr_str][region.start-1-n_per_side : region.stop+n_per_side]
		
		assert pre_seq.seq + str_seq.seq + post_seq.seq == full_seq.seq
		samples.append({
			'HipSTR_name': region.str_name,
			'motif': region.motif,
			'motif_len': region.motif_len,
			'num_copies': region.num_copies,
			'str_seq': str_seq.seq,
			'str_seq_name': str_seq.fancy_name,
			'pre_seq': pre_seq.seq,
			'pre_seq_name': pre_seq.fancy_name,
			'post_seq': post_seq.seq,
			'post_seq_name': post_seq.fancy_name,
			'full_seq': full_seq.seq,
			'full_seq_name': full_seq.fancy_name,
			'n_per_side': n_per_side
		})

	# Save unlabeled samples to main data dict
	this_sample_set_fname = 'unlabeled_samples_GRCh38_{}_per_side.json'.format(n_per_side)
	samples_save_path = os.path.join('..', 'data', 'unlabeled_samples', 
										this_sample_set_fname)

	with open(samples_save_path, 'w') as fp:
		json.dump(samples, fp, indent=4)
