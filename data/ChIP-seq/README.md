Sample GSM3579716 data from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi

More possibly at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125660

Install bigWigToBedGraph from UCSC using:
```
conda install -c bioconda ucsc-bigwigtobedgraph
```

Convert bigwig (.bw) format ChIP-seq files to bedgraph using:
```
bigWigToBedGraph ChIP_fname.bw ChIP_fname.bedgraph
```

Peak call
```
macs3 bdgpeakcall -i ChIP_fname.bedgraph -o ChIP_fname.bed
```