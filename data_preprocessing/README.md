# Reapeat Number Variation Data Preprocessing Pipeline
- fine_target_STRs.py - Extracts STRs of the desired motiff from a hipstr_reference.bed.gz for the desired chromosome build.

- label_STRs_heterozygosity.py - Labels each STR with data from CEU_filtered_strhet.tab, then outputs labeled_samples_het.json

- preprocessV2_repeat_var.py - Filter by STR type, repeat consistency, max STR len, min num called. Saves to sample_data_V2_repeat_var.json to use with dataloaders