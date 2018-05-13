# PCA for Infering Genetic Ancestry:

Coded by Jérémie KALFON in Python2 + vcftools for the BroadInstitute

## Instalation

you should have python2 installed and Anaconda

you can import the repo with git and the git command

```bash
git clone [url] 
cd PCA-Ancestry
pip install -r requirements.txt
```

now you can launch a jupyter notebook or python
and execute the commands

## What you can do 

## TODO

- use IPCA 
- try KPCA
- do CV 
- do other accuracy tests
- do the simple visualisation 
- add tsne visualisation 
- add requirements 
- add a cmd line function 
- add another clustering method (maybe using autosklearn, without PCA.)
- make the code compatible python3 with `future`

## Problems and ideas

We could try to look for other feature selection method as PCA is often not recommended for classification problems
- ACO 
- Dictionnary learning with GAs
- Conv Neural Nets if enough training data

We could try gaussian processes for automatic selection of params in vcftools
We could use Naive Bayes to guess missing values



Jérémie KALFON for BroadInstitute
jkobject@gmail.com
jkobject.com



