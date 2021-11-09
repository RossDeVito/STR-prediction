import torch
import torch.nn as nn
import torch.nn.functional as F

from dnabert.transformers.configuration_bert import BertConfig
from dnabert.transformers.modeling_bert import BertForSequenceClassification
from dnabert.transformers.tokenization_bert import BertTokenizer


class DNABERT(nn.Module):
	""" Loads pretrained DNABERT model as feature extractor. """
	def __init__(self, save_path='dnabert/5-new-12w-0/'):
		super().__init__()
		self.bert_config = BertConfig.from_pretrained(
			'dnabert/5-new-12w-0/',
			num_labels=2
		)
		self.model = BertForSequenceClassification.from_pretrained(
			'dnabert/5-new-12w-0/',
			config=self.bert_config,
			from_tf=False,
		)
		self.model = self.model.bert

	def forward(self, x):
		if x['input_ids'].dim() == 1:
			return self.model(
				input_ids=x['input_ids'].unsqueeze(0),
				token_type_ids=x['token_type_ids'].unsqueeze(0),
				attention_mask=x['attention_mask'].unsqueeze(0),
			)
		else:
			return self.model(**x)


if __name__ == '__main__':
	import os
	from data_modules import STRHetPrePostDataModule

	# Load BERT tokenizer
	tokenizer = BertTokenizer.from_pretrained(
		'dnabert/5-new-12w-0/',
		from_tf=False,
		do_lower_case=False
	)

	# STR het prepost
	data_dir = os.path.join('..', 'data', 'heterozygosity', 'samples_prepost_2')
	split_file = 'split_1.json'

	STR_data = STRHetPrePostDataModule(
		data_dir, 
		split_file, 
		num_workers=2, 
		n_per_side=500,
		tokenizer=tokenizer,
		k=5	
	)
	STR_data.setup()

	fe = DNABERT()

	# Test
	batch = next(iter(STR_data.val_dataloader()))
	d = STR_data.val_data[0]

	x_pre = {
		'input_ids': d['pre_input_ids'],
		'token_type_ids': d['pre_token_type_ids'],
		'attention_mask': d['pre_attention_mask']
	}
	x_pre_bat = {
		'input_ids': batch['pre_input_ids'],
		'token_type_ids': batch['pre_token_type_ids'],
		'attention_mask': batch['pre_attention_mask']
	}

	r_d = fe(x_pre)
	r_bat = fe(x_pre_bat)

	torch.cat([r_bat[1], r_bat[1]], -1).shape
