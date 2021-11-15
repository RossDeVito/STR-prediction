"""Creates input feature representation for labeled samples, finds
corresponding feature string and chromosome range, removes STRs with
length over threshold, and saves resulting samples in a JSON fine
"""

import json
import os
import sys
import math

import numpy as np
import pandas as pd
from tqdm import tqdm


def make_feat_mat(seq_string, distance):
	bases = np.array(list(seq_string))
	unk = (bases == 'N').astype(int) / 4

	A = unk + (bases == 'A')
	C = unk + (bases == 'C')
	G = unk + (bases == 'G')
	T = unk + (bases == 'T')

	return np.vstack((A, C, G, T, distance))


def make_pre_post_features(samp_dict, n_per_side=500, bp_dist_base=1000.0):
	"""Given a dictionary representing a STR sample (like those generated
	by find_target_STRs.py or label_STRs.py), creates feature matrix
	where first n_per_side indices are for pre-str sequence and the 
	remaining n_per_side the post-str seqence.

	Returns: Matrix with features A, C, G, T, distance_from_str
		A, C, G, T: one-hot encoding of sequence (or all .25 if unknown "N")
		distance_from_str: distance from STR in kBp
	"""
	pre_seq = samp_dict['pre_seq']
	post_seq = samp_dict['post_seq']

	# If sample from complement strand, reverse seqence direction
	if samp_dict['complement'] == True:
		new_pre_seq = post_seq[::-1]
		post_seq = pre_seq[::-1]
		pre_seq = new_pre_seq

	pre_seq = pre_seq[-n_per_side:]
	post_seq = post_seq[:n_per_side]
	assert len(pre_seq) == len(post_seq) == n_per_side

	seq_string = pre_seq + post_seq
	
	dists = np.hstack((
		np.array(list(range(n_per_side, 0, -1))) * -1 / bp_dist_base,
		np.array(list(range(1,n_per_side+1, 1))) / bp_dist_base
	))

	# Create string describing new seqence's location
	str_seq_name = samp_dict['str_seq_name']
	chr_name, loc_range = str_seq_name.split(':')
	start, end = (int(v) for v in loc_range.split('-'))
	
	if samp_dict['complement'] == True:
		new_name = '{}:{}-{} (complement)'.format(chr_name, start-n_per_side, end+n_per_side)
	else:
		new_name = '{}:{}-{}'.format(chr_name, start-n_per_side, end+n_per_side)

	# Create feature matrix
	seq_feat_mat = make_feat_mat(seq_string, dists)

	return seq_feat_mat, seq_string, new_name


if __name__ == '__main__':
	max_STR_len = 100
	n_per_side = 500

	min_num_called = None # if None will skip, for het task

	# Load labeled STRs to be preprocessed
	samp_dir = os.path.join('..', 'data', 'mecp2_binding')
	samp_fname = 'labeled_samples_mecp2.json'
	samp_path = os.path.join(samp_dir, samp_fname)

	with open(samp_path) as fp:    
		samples = json.load(fp)

	sample_data = []
	labels = []

	save_dir = os.path.join('..', 'data', 'mecp2_binding', 'samples_pp')

	# Filter samples by STR length, then create formatted output_seq_len samples
	for i in tqdm(range(len(samples)), file=sys.stdout):
		samp_dict = samples.pop(0)

		# filter out by STR length
		if samp_dict['motif_len'] * samp_dict['num_copies'] > max_STR_len:
			continue

		# filter out by min num called
		if min_num_called is not None and samp_dict['num_called'] < min_num_called:
			continue

		seq_mat, seq_string, chr_loc = make_pre_post_features(samp_dict, n_per_side)
			
		# Save feature matrix to disk
		fm_fname = "{}.npy".format(i)
		np.save(os.path.join(save_dir, fm_fname), seq_mat)

		sample_data.append({
			'fname': fm_fname,
			'seq_string': seq_string,
			'chr_loc': chr_loc,
			'label': samp_dict['label'],
			'HipSTR_name': samp_dict['HipSTR_name']
		})

		# # for dev
		# if i > 10000:
		# 	break

	# Print stats
	labels = np.array(labels)
	print("Total samples:\t{}".format(len(labels)))
	print("\t0:\t{}".format(((labels == 0).sum())))
	print("\t1:\t{}".format(((labels > 0).sum())))

	# Save JSON of preprocessed samples
	this_sample_set_fname = 'sample_data.json'
	save_path = os.path.join(save_dir, this_sample_set_fname)

	with open(save_path, 'w') as fp:
		json.dump(sample_data, fp, indent=4)