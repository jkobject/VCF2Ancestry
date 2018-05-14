### Standard Pipeline
#!bin/sh
vcftools --gzvcf $name --recode --out data/lowDPbefore --maf 0.01 --min-alleles 2 --max-alleles 2 --max-missing 0.8
vcftools --vcf data/lowDPbefore.recode.vcf --missing-indv --out data/out
mawk '$4 > 30000' data/out.imiss | cut -f1 > data/lowDP.indv
vcftools --vcf 'data/lowDPbefore.recode.vcf' --recode --remove data/lowDP.indv --out data/filtered2
rm data/lowDP*
mkdir data/chunks
for i in {1..22} do; vcftools  --vcf  data/filtered2.recode.vcf  --chr $i --recode --recode-INFO-all --out  data/chunks/VCF_ch$i end
vcftools  --vcf  data/filtered2.recode.vcf  --chr MT --recode --recode-INFO-all --out  data/chunks/VCF_chMT
vcftools  --vcf  data/filtered2.recode.vcf  --chr X --recode --recode-INFO-all --out  data/chunks/VCF_chX
vcftools  --vcf  data/filtered2.recode.vcf  --chr Y --recode --recode-INFO-all --out  data/chunks/VCF_chY
for i in {1..22} do; vcftools --vcf data/chunks/VCF_ch$i.recode.vcf --min-r2 0.1 --geno-r2 --out data/chunks/filtVCF_ch$i & end
vcftools --vcf data/chunks/VCF_chMT.recode.vcf --min-r2 0.1 --geno-r2 --out data/chunks/filtVCF_chMT &
vcftools --vcf data/chunks/VCF_chX.recode.vcf --min-r2 0.1 --geno-r2 --out data/chunks/filtVCF_chX &            
vcftools --vcf data/chunks/VCF_chY.recode.vcf --min-r2 0.1 --geno-r2 --out data/chunks/filtVCF_chY &
await
rm data/*.log
cat data/chunks/filtVCF_ch* > data/all_VCF.geno.lg
vcftools --vcf data/filtered2.recode.vcf --exclude-positions data/all_VCF.geno.lg --recode --out data/$out