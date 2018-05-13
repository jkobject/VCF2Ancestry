# source code for the program
# Made by Jeremie KALFON in May 2018


# A set of principal component values for the call set
# (i.e., a table containing the PC values for each sample for each principal component)

# A final classification or ancestry label assigned to each sample that is missing a label

# A visualization of the distribution of PC values for each sample in the call set,
# along with the labeled and predicted ancestry classifications

import os
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import vcf
from sklearn.model_selection import train_test_split
import psutil
from sklearn.model_selection import cross_val_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import time


class Population(object):
    """docstring for Population"""

    types = {'eas': 'east asian',
             'nfe': 'non finish european',
             'sas': 'south asian',
             'afr': 'african',
             'amr': 'mixed american'}
    maxallelefreq = None
    callrate = None
    outfile = None
    labeled = None
    nbcomp = None
    valofinterest = None
    clf = None
    train_pha = None
    test_pha = None
    train_gt = None
    test_gt = None
    train_red = None
    pred_red = None
    rec = None
    classifier = None
    tofind = None

    def __init__(self, arg):
        super(Population, self).__init__()
        self.arg = arg

    def __getitem__(self, key):
        print self.types[list(self.labeled.loc[self.labeled['sample_id'] == key]['ancestry'])[0]]

    def __len__(self):
        return self.labeled.shape[0], self.tofind.shape[0]

    def __iter__(self, labeled=True):
        return iter(list(self.labeled['sample_id']))

# Utilities

    def _get_training_data(self, inp, percentage=0.3):
        X_train, X_test, y_train, y_test = train_test_split(inp, self.labeled['ancestry'], test_size=percentage, random_state=0)
        return X_train, X_test, y_train, y_test

# Functions

    def load_dataset(self, filename):
        pass

    def filter_variants(self, name="data/mini_project.vcf.bgz", out="out", minmissing=30000, maxr2=0.01, callrate=0.8, maxallelefreq=0.01):
        """
        Successful, clean PCA on human genetic data will require
        filtering data to high-quality variants that are linkage disequilibrium (LD)-pruned.
        In general, we like to run PCA on high-callrate, bi-allelic,
        common (allele frequency > 0.01) variants that are pruned to r^2<0.1;

        but you are welcome to run PCA on whichever set of variants you find work best for you.
        min missing r2

        """
        self.maxallelefreq = maxallelefreq
        self.callrate = callrate
        self.outfile = out
        filt = "vcftools --vcf '" + name + "' --recode --out lowDPbefore"
        filt += " --maf " + maxallelefreq
        filt += ' --min-alleles 2 --max-alleles 2 '
        filt += ' --max-missing ' + callrate
        print "applying first filter"
        os.system(filt)
        print "applying first filter"
        cmd = 'vcftools --vcf "trial1.recode.vcf" --missing-indv'
        os.system(cmd)
        print "finding if too much missing individuals and recoding the file"
        os.system("mawk '$4 > 30000' out.imiss | cut - f1 > lowDP.indv")
        os.system("vcftools --vcf 'lowDPbefore.recode.vcf' --recode --remove lowDP.indv --out filtered2")
        print "removing garbage.."
        os.system('rm lowDP*')

        vcf_reader = vcf.Reader(open('data/filtered2.recode.vcf', 'r'))
        print "dividing the input file.."
        for i in vcf_reader.contigs.keys():
            i = str(i)
            if len(i) < 3:
                os.system("vcftools  --vcf  filtered2.recode.vcf  --chr " + i +
                          " --recode --recode-INFO-all --out  chunks/VCF_ch" + i)
        print "running the ld prunning in parallel (might still take time (avg is 60mn)"
        for i in vcf_reader.contigs.keys():
            i = str(i)
            if len(i) < 3:
                os.system("vcftools --vcf chunks/VCF_" + i +
                          ".recode.vcf --min-r2 0.1 --geno-r2 --out chunks/VCF_" + i + " &")
        start = time.timer()
        while(True):
            nbjob = 0
            for p in psutil.process_iter():
                try:
                    if str(p.name()) == 'vcftools':
                        nbjob += 1
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                except psutil.NoSuchProcess:
                    continue
            if nbjob == 0:
                break
            else:
                print "there is still" + str(nbjob) + " jobs \r",
        end = time.timer()
        print "it took " + str(end - start) + " seconds"
        print "concatenating every file"
        os.system('rm *.log')
        os.system('cat VCF_* > all_VCF.geno.lg')
        print "now prunning..."
        os.system('vcftools --vcf filtered2.recode.vcf --exclude-positions all_VCF.geno.lg --recode --out' + out)

    def reducedim(self, inp=None, topred=None, reducer='pca', n_components=500, val='gt'):
        self.nbcomp = n_components
        self.valofinterest = val
        if inp is None:
            inp = self.train_gt if val is 'gt' else self.train_pha
        if topred is None:
            topred = self.test_gt if val is 'gt' else self.test_pha
        toreduce = np.vstack(inp, topred)
        if reducer is 'pca':
            redu = PCA(n_components=n_components)
        red = redu.fit_transform(toreduce)
        self.train_red = red[:inp.shape[0]]
        self.pred_red = red[inp.shape[0]:]

    def train_classifier(self, inp=None, classifier='knn', test='CV', scoring='accuracy', percentage=0.3):
        if inp is None:
            inp = self.train_red
        self.classifier = classifier
        if classifier is 'adaboost':
            self.clf = AdaBoostClassifier(n_estimators=self.nbcomp * 0.7)
        if classifier is 'knn':
            self.clf = NearestCentroid()
        if test is 'CV':
            scores = cross_val_score(self.clf, inp, self.labeled['ancestry'], scoring=scoring, cv=3, n_jobs=1)
            print "cv scores : " + scores
            score = np.mean(scores)
        if test is 'reg':
            X_train, X_test, y_train, y_test = self.get_training_data(inp, percentage=0.3)
            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
        print "the total score is of " + score

    def predict_labels(self, inp):
        if self.clf is not None:
            return self.clf.predict(inp) if inp is not None else self.clf.predict(self.pred_red)

    def compute_features_nb(self, vmin=50, vmax=1000):
        for val in range(vmin, vmax):
            pass

    def extract_unlabeled(self):
        labels = pd.read_csv("data/acb_ii_mini_project_labels.txt", sep='\t')
        indices = labels['ancestry'].isna()
        self.tofind = labels[indices]
        self.labeled = labels[indices is False]

    def load_from_vcf(self, filename='data/out.recode.vcf', printinfo=True, maxval=1000, keep_prev=False):
        """
        parloadval read from the filtered vcf file, the names given in df

        Params:
        ------
        df : dataframe - a dataframe with a sample_id index containing the names
        of the different samples to extract from the file

        filename : str - the name of the file

        printinfo : flag - show the information about the vcf file being read

        Returns:
        -------
        gt: np.array [nbofindividuals,nbofrecords] - 0-1-2 values stating if the genotype has
        the ALTval in 0-1-2 of its chromosomes.

        pha: np.array [nbofindividuals,nbofrecords] - bool stating if this variant is phased or not

        rec: dict[chromvalue:list[POS,REFval,ALTval]] - a dicionnary of meta information about the
        records being read

        """
        vcf_reader = vcf.Reader(open('data/trial1.recode.vcf', 'r'))
        if printinfo:
            print "having " + str(len(vcf_reader.contigs)) + " chromosomes"
            size = 0
            for key, val in vcf_reader.contigs.iteritems():
                size += val.length
            print vcf_reader.metadata
            print size
        label_names = list(self.labeled['sample_id'])
        test_names = list(self.tofin['sample_id'])
        if not keep_prev:
            self.rec = {}
            self.train_gt = []
            self.train_pha = []
            self.test_gt = []
            self.test_pha = []
        else:
            self.train_gt = self.train_gt.tolist()
            self.train_pha = self.train_pha.tolist()
            self.test_gt = self.test_gt.tolist()
            self.test_pha = self.test_pha.tolist()

        train_gt = []
        train_pha = []
        test_gt = []
        test_pha = []
        """
        # A version in parallel but is finally slower than this one (not optimized enough..)
        def inpar(record, names):
            print "doing position : " + str(record.POS)
            gt = []
            pha = []
            rec = [record.CHROM, record.POS, record.REF, record.ALT]
            for name in names:
                gt.append(record.genotype(name).gt_type)
                pha.append(record.genotype(name).phased)
            return [gt, pha, rec]
        values = Parallel(n_jobs=-1)(delayed(inpar)(record, names) for record in vcf_reader)
        for val in values:
            gt.append(val[0])
            pha.append(val[1])
            rec.append(val[2])
        """
        has = False
        chrom = -1
        for i, record in enumerate(vcf_reader):
            if keep_prev and self.rec[record.CHROM] is not None:
                if self.rec[record.CHROM][record.POS] is not None:
                    has = True
            if not has:
                print "doing chrom,pos : " + str(record.CHROM) + ',' + str(record.POS) + "\r",
                if record.CHROM != chrom:
                    chrom = record.CHROM
                    self.rec.update({chrom: {}})
                self.rec[chrom].update({record.POS: [record.REF, record.ALT]})
                for name in label_names:
                    train_gt.append(record.genotype(name).gt_type)
                    train_pha.append(record.genotype(name).phased)
                for name in test_names:
                    test_gt.append(record.genotype(name).gt_type)
                    test_pha.append(record.genotype(name).phased)
                self.train_gt.append(train_gt)
                self.train_pha.append(train_pha)
                self.test_gt.append(test_gt)
                self.test_pha.append(test_pha)
            if i >= maxval:
                break
        # """
        # we are using numpy, more efficient
        # we order by individuals x records
        self.test_pha = np.array(self.test_pha, dtype=np.bool).T
        self.test_gt = np.array(self.test_gt).T
        self.train_pha = np.array(self.train_pha, dtype=np.bool).T
        self.train_gt = np.array(self.train_gt).T
        print "PHASE nonzero " + str(np.count_nonzero(train_pha))
        print "SNPs nonzero " + str(np.count_nonzero(train_gt))
        for key, val in self.types:
            print "you have " + self.labeled.loc[self.labeled['ancestry'] == key].shape[0] + " " + val + "in your labeled set"
        print "we have a dataset of "
