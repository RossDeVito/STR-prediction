"""Creates input feature representation for labeled samples, finds
corresponding feature string and chromosome range, removes STRs with
length over threshold, and saves resulting samples in a JSON fine
"""

import json
import os
import math

import numpy as np
import pandas as pd
from tqdm import tqdm


def make_feat_mat(seq_string, is_STR, STR_dist, distance):
	bases = np.array(list(seq_string))
	unk = (bases == 'N').astype(int) / 4

	A = unk + (bases == 'A')
	C = unk + (bases == 'C')
	G = unk + (bases == 'G')
	T = unk + (bases == 'T')

	return np.vstack((A, C, G, T, is_STR, STR_dist, distance))


def make_features(samp_dict, output_seq_len, bp_dist_base=1000.0):
	"""Given a dictionary representing a STR sample (like those generated
	by find_target_STRs.py or label_STRs.py), finds the input_seq_len
	sequence centered at the STR. For this centered sequence, genereates a
	feature matrix (which will be input for models), sequence string,
	and chromosome location.
	"""
	str_seq = samp_dict['str_seq']
	str_seq_len = len(str_seq)
	
	n_surrounding = output_seq_len - str_seq_len
	n_before = math.ceil(n_surrounding / 2.0)
	n_after = math.floor(n_surrounding / 2.0)

	pre_seq = samp_dict['pre_seq']
	post_seq = samp_dict['post_seq']

	# If sample from complement strand, reverse seqence direction
	if samp_dict['complement'] == True:
		str_seq = str_seq[::-1]
		new_pre_seq = post_seq[::-1]
		post_seq = pre_seq[::-1]
		pre_seq = new_pre_seq

	pre_seq = pre_seq[-n_before:]
	post_seq = post_seq[:n_after]

	seq_string = pre_seq + str_seq + post_seq
	is_STR = np.hstack(
		(np.zeros(n_before), np.ones(str_seq_len), np.zeros(n_after))
	)
	STR_dist = np.hstack((
		-np.linspace(n_before, 1, n_before) / 1000, # units of kBp
		np.zeros(str_seq_len),
		np.linspace(1, n_after, n_after) / 1000
	))
	distance = np.linspace(0, output_seq_len/1000.0, output_seq_len, 
							endpoint=False)
	distance = distance - distance.mean()

	assert len(seq_string) == output_seq_len

	# Create string describing new seqence's location
	str_seq_name = samp_dict['str_seq_name']
	chr_name, loc_range = str_seq_name.split(':')
	start, end = (int(v) for v in loc_range.split('-'))
	
	if samp_dict['complement'] == True:
		new_name = '{}:{}-{} (complement)'.format(chr_name, start-n_after, end+n_before)
	else:
		new_name = '{}:{}-{}'.format(chr_name, start-n_before, end+n_after)

	# Create feature matrix
	seq_feat_mat = make_feat_mat(seq_string, is_STR, STR_dist, distance)

	return seq_feat_mat, seq_string, new_name


if __name__ == '__main__':
	max_STR_len = 100
	output_seq_len = 1000

	min_num_called = 100 # if None will skip, for het task

	# Load labeled STRs to be preprocessed
	samp_dir = os.path.join('..', 'data', 'heterozygosity')
	samp_fname = 'labeled_samples_het.json'
	samp_path = os.path.join(samp_dir, samp_fname)

	with open(samp_path) as fp:    
		samples = json.load(fp)

	sample_data = []
	labels = []

	save_dir = os.path.join('..', 'data', 'heterozygosity', 'samples2')

	# Filter samples by STR length, then create formatted output_seq_len samples
	for _ in tqdm(range(len(samples))):
		samp_dict = samples.pop(0)

		# filter out by STR length
		if samp_dict['motif_len'] * samp_dict['num_copies'] > max_STR_len:
			samp_dict = None
			del samp_dict
			continue

		# filter out by min num called
		if min_num_called is not None and samp_dict['num_called'] < min_num_called:
			samp_dict = None
			del samp_dict
			continue

		seq_mat, seq_string, chr_loc = make_features(samp_dict, output_seq_len)
		
		# Save feature matrix to disk
		fm_fname = "{}.npy".format(len(sample_data))
		np.save(os.path.join(save_dir, fm_fname), seq_mat)

		sample_data.append({
			'fname': fm_fname,
			'seq_string': seq_string,
			'chr_loc': chr_loc,
			'label': samp_dict['label'],
			'HipSTR_name': samp_dict['HipSTR_name']
		})

		labels.append(samp_dict['label'])
		samp_dict = None
		del samp_dict

		# for dev
		if len(labels) > 10000:
			break

	# Print stats
	labels = np.array(labels)
	print("Total samples:\t{}".format(len(labels)))
	print("\t0:\t{}".format(((labels == 0).sum())))
	print("\t1:\t{}".format(((labels == 1).sum())))

	# Save JSON of preprocessed samples
	this_sample_set_fname = 'sample_data.json'
	save_path = os.path.join(save_dir, this_sample_set_fname)

	with open(save_path, 'w') as fp:
		json.dump(sample_data, fp, indent=4)