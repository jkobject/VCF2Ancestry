# source code for the program
# Made by Jeremie KALFON in May 2018


# A set of principal component values for the call set
# (i.e., a table containing the PC values for each sample for each principal component)

# A final classification or ancestry label assigned to each sample that is missing a label

# A visualization of the distribution of PC values for each sample in the call set,
# along with the labeled and predicted ancestry classifications

import os
from itertools import chain
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA, KernelPCA
import pandas as pd
import vcf
from sklearn.model_selection import train_test_split
import psutil
from sklearn.model_selection import cross_val_score
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier as GCP
import time
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
try:
    from urllib2 import urlopen as urlopen
except ImportError:
    from urllib.request import urlopen as urlopen


class Population(object):
    """docstring for Population"""

    types = {'eas': 'east asian',
             'nfe': 'non finish european',
             'sas': 'south asian',
             'afr': 'african',
             'amr': 'mixed american',
             'nan': 'unknown',
             'fin': 'finish'}

    dataset = {'labels_miniproj.txt': 'https://www.dropbox.com/s/dmgchsjklm1jvgk/\
    acb_ii_mini_project_labels.txt?dl=1',
               'data_miniproj.vcf.bgz': 'https://www.dropbox.com/s/iq8c81awi31067c/\
    acb_ii_mini_project.vcf.bgz?dl=1'}
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
    found = None

    def __init__(self):
        super(Population, self).__init__()
        os.system('mkdir data')

    def __getitem__(self, key):
        print self.types[list(self.all.loc[self.all['sample_id'] == key]['ancestry'])[0]]

    def __len__(self):
        return (self.labeled.shape[0], self.tofind.shape[0]) if self.labeled is not None else 0

    def __iter__(self, labeled=True):
        return iter(list(self.labeled['sample_id']))

    def keys(self):
        return list(self.all['sample_id']).__iter__()

    def values(self):
        return list(self.all['ancestry']).__iter__()

# Utilities

    def _get_training_data(self, inp, percentage=0.3):
        X_train, X_test, y_train, y_test = train_test_split(inp, self.labeled['ancestry'],
                                                            test_size=percentage, random_state=0)
        return X_train, X_test, y_train, y_test

# Functions

    def load_dataset(self, filename=None, url=None):
        if filename is None:
            for key, val in self.dataset.iteritems():
                if not os.path.exists('data/' + key):
                    print "downloading " + key + " with urllib"
                    f = urlopen(val)
                    data = f.read()
                    with open('data/' + key, "wb") as code:
                        code.write(data)
                else:
                    print "file is already there"
        else:
            if not os.path.exists(filename):
                print "downloading " + filename + " with urllib"
                f = urlopen(url)
                data = f.read()
                with open(filename, "wb") as code:
                    code.write(data)
            else:
                print "file is already there"

    def filter_variants(self, name="data/data_miniproj.vcf.bgz", out="out",
                        minmissing=30000, maxr2=0.01, callrate=0.8, maxallelefreq=0.01):
        """
        Successful, clean PCA on human genetic data will require
        filtering data to high-quality variants that are linkage disequilibrium (LD)-pruned.
        In general, we like to run PCA on high-callrate, bi-allelic,
        common (allele frequency > 0.01) variants that are pruned to r^2<0.1;

        but you are welcome to run PCA on whichever set of variants you find work best for you.
        min missing r2

        """
        print "assuming you have mawk, vcftools, cat, cut installed"
        self.maxallelefreq = maxallelefreq
        self.callrate = callrate
        self.outfile = out
        filt = "vcftools --gzvcf '" + name + "' --recode --out data/lowDPbefore"
        filt += " --maf " + str(maxallelefreq)
        filt += ' --min-alleles 2 --max-alleles 2'
        filt += ' --max-missing ' + str(callrate)
        print "applying first filter"
        os.system(filt)
        print "applying second filter"
        os.system('vcftools --vcf "data/lowDPbefore.recode.vcf" --missing-indv --out data/out')
        print "finding if too much missing individuals and recoding the file"
        os.system("mawk '$4 > 30000' data/out.imiss | cut -f1 > data/lowDP.indv")
        os.system("vcftools --vcf 'data/lowDPbefore.recode.vcf' --recode --remove data/lowDP.indv\
         --out data/filtered2")
        print "removing garbage.."
        os.system('rm data/lowDP*')

        vcf_reader = vcf.Reader(open('data/filtered2.recode.vcf', 'r'))
        os.system('mkdir data/chunks')
        print "dividing the input file.."
        for i in vcf_reader.contigs.keys():
            i = str(i)
            if len(i) < 3:
                os.system("vcftools  --vcf  data/filtered2.recode.vcf  --chr " + i +
                          " --recode --recode-INFO-all --out  data/chunks/VCF_ch" + i)
        print "running the ld prunning in parallel (might still take time (avg is 60mn)"
        for i in vcf_reader.contigs.keys():
            i = str(i)
            if len(i) < 3:
                os.system("vcftools --vcf data/chunks/VCF_ch" + i +
                          ".recode.vcf --min-r2 0.1 --geno-r2 --out data/chunks/filtVCF_ch" +
                          i + " &")
        start = time.time()
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
                print "there is still " + str(nbjob) + " jobs \r",
        end = time.time()
        print "it took " + str(end - start) + " seconds"
        print "concatenating every file"
        os.system('rm data/*.log')
        os.system('cat data/chunks/filtVCF_ch* > data/all_VCF.geno.lg')
        print "now prunning..."
        os.system('vcftools --vcf data/filtered2.recode.vcf --exclude-positions \
            data/all_VCF.geno.lg --recode --out data/' + out)

    def extract_unlabeled(self, filename=None):
        filename = filename if filename is not None else "data/labels_miniproj.txt"
        labels = pd.read_csv(filename, sep='\t')
        indices = labels['ancestry'].isna()
        self.tofind = labels[indices]
        self.labeled = labels[indices == False]
        self.all = pd.concat([self.labeled, self.tofind])

    def load_from_vcf(self, filename=None, printinfo=True, maxval=1000, keep_prev=False):
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
        filename = filename if filename is not None else 'data/' + self.outfile + '.recode.vcf'
        vcf_reader = vcf.Reader(open(filename, 'r'))
        if printinfo:
            print "having " + str(len(vcf_reader.contigs)) + " chromosomes"
            size = 0
            for key, val in vcf_reader.contigs.iteritems():
                size += val.length
            print "meta :"
            print vcf_reader.metadata
            print "genomesize : "
            print size
        label_names = list(self.labeled['sample_id'])
        test_names = list(self.tofind['sample_id'])
        if not keep_prev:
            self.train_gt = np.empty((0, len(label_names)), int)
            self.train_pha = np.empty((0, len(label_names)), bool)
            self.test_gt = np.empty((0, len(test_names)), int)
            self.test_pha = np.empty((0, len(test_names)), bool)
            self.rec = {}
        else:
            self.test_pha = self.test_pha.T
            self.test_gt = self.test_gt.T
            self.train_pha = self.train_pha.T
            self.train_gt = self.train_gt.T
        chrom = -1
        j = 0
        numa = 0
        count = 0
        for record in vcf_reader:
            if keep_prev:
                for key, val in self.rec.iteritems():
                    for key, val2 in val.iteritems():
                        vcf_reader.next()
                        numa += 1
                keep_prev = False
            count = numa + j
            print "doing chrom : " + str(record.CHROM) + ', at pos : ' + str(record.POS)\
                + " ,number : " + str(count) + "\r",
            if record.CHROM != chrom:
                chrom = record.CHROM
                if record.CHROM not in self.rec:
                    self.rec.update({chrom: {}})
            self.rec[chrom].update({record.POS: [record.REF, record.ALT]})
            train_gt = np.zeros(len(label_names))
            train_pha = np.zeros(len(label_names))
            for i, name in enumerate(label_names):
                train_gt[i] = record.genotype(
                    name).gt_type if record.genotype(name).gt_type is not None else 0
                train_pha[i] = record.genotype(
                    name).phased if record.genotype(name).phased is not None else 0

            test_gt = np.zeros(len(test_names))
            test_pha = np.zeros(len(test_names))
            for i, name in enumerate(test_names):
                test_gt[i] = record.genotype(
                    name).gt_type if record.genotype(name).gt_type is not None else 0
                test_pha[i] = record.genotype(
                    name).phased if record.genotype(name).phased is not None else 0
            self.train_gt = np.vstack((self.train_gt, train_gt))
            self.train_pha = np.vstack((self.train_pha, train_pha))
            self.test_gt = np.vstack((self.test_gt, test_gt))
            self.test_pha = np.vstack((self.test_pha, test_pha))
            j += 1
            if j > maxval - 1:
                break
        # """
        # we are using numpy, more efficient
        # we order by individuals x records
        self.test_pha = self.test_pha.T
        self.test_gt = self.test_gt.T
        self.train_pha = self.train_pha.T
        self.train_gt = self.train_gt.T
        print ' '  # to jump a line
        print "PHASE nonzero " + str(np.count_nonzero(self.train_pha))
        print "SNPs nonzero " + str(np.count_nonzero(self.train_gt))
        for key, val in self.types.iteritems():
            print "you have " + str(self.labeled.loc[self.labeled['ancestry'] == key].shape[0])\
                + " " + str(val) + "in your labeled set"

    def par_load_from_vcf(self, filename, printinfo=True):
        """
        the parallel version of loadfromvcf,should be way faster

        same inputs but reduced choice for now
        """
        filename = filename if filename is not None else 'data/' + self.outfile + '.recode.vcf'
        vcf_reader = vcf.Reader(open(filename, 'r'))
        print "dividing the input file.."
        files = []
        for i in vcf_reader.contigs.keys():
            i = str(i)
            if len(i) < 3:
                files.append(i)
                # os.system("vcftools  --vcf  " + filename + " --chr " + i +
                #         " --recode --recode-INFO-all --out  data/chunks/inpar_ch" + i)
        label_names = list(self.labeled['sample_id'])
        test_names = list(self.tofind['sample_id'])
        self.rec = {}
        self.train_gt = np.empty((0, len(label_names)), int)
        self.train_pha = np.empty((0, len(label_names)), bool)
        self.test_gt = np.empty((0, len(test_names)), int)
        self.test_pha = np.empty((0, len(test_names)), bool)
        if printinfo:
            print "having " + str(len(vcf_reader.contigs)) + " chromosomes"
            size = 0
            for key, val in vcf_reader.contigs.iteritems():
                size += val.length
            print vcf_reader.metadata
            print size
        print "prints everythig outside of the notebook (stdout problem by the guys of joblib)"
        print "should be resolved in the next release"
        vals = Parallel(n_jobs=-1)(delayed(_inpar)(file, label_names, test_names) for file in files)
        for i, val in enumerate(vals):
            if len(val[1]) != 0:
                # wether or not it is equal to zero we consider it is the same for all others
                self.train_gt = np.vstack((self.train_gt, convertlist(val[1])))
                self.train_pha = np.vstack((self.train_pha, convertlist(val[2], type=np.bool)))
                self.test_gt = np.vstack((self.test_gt, convertlist(val[3])))
                self.test_pha = np.vstack((self.test_pha, convertlist(val[4], type=np.bool)))
            self.rec.update({files[i]: val[0]})
        self.test_pha = self.test_pha.T
        self.test_gt = self.test_gt.T
        self.train_pha = self.train_pha.T
        self.train_gt = self.train_gt.T
        print "PHASE nonzero " + str(np.count_nonzero(self.train_pha))
        print "SNPs nonzero " + str(np.count_nonzero(self.train_gt))
        for key, val in self.types.iteritems():
            print "you have " + str(self.labeled.loc[self.labeled['ancestry'] == key].shape[0])
            + " " + str(val) + " in your labeled set"
        os.system("rm *.log")
        os.system("rm data/*.log")
        os.system("rm data/chunks/*.log")

    def reduce_features(self, inp=None, topred=None, reducer='pca', n_components=500, val='gt',
                        retrain=True):
        """
        will use a dimensionality reduction algorithm to reduce the number of features of the dataset

        you can pass it you own inputs or use the ones that are stored in the file

        Params:
        ------
        inp: np.array[values,features],the input array you have and want to reduce and will train on
        topred: np.array[values,features], the input array you have and want to reduce and predict
        reducer: str, the reducer algorithm to use (pca,)
        n_components : int, the final number of features in your reduced dataset
        val : str (gt|pha), to see if there is any predictibility using phasing..
        retrain: flag, set to false if you already have trained the PCA and don't want to restart
        (espacially important if you consider to compare two different datasets)

        Outs:
        ----
        nbcomp: saves the number of components
        valofinterest: the value of interest (val)
        train_red, pred_red: and the reduced train and pred arrays 
        """
        self.nbcomp = n_components
        self.valofinterest = val
        if inp is None:
            inp = self.train_gt if val is 'gt' else self.train_pha
        if topred is None and inp is None:
            topred = self.test_gt if val is 'gt' else self.test_pha
        toreduce = np.vstack((inp, topred)) if topred is not None else inp
        if reducer is 'pca':
            redu = PCA(n_components=n_components)
        if reducer is 'kpca':
            redu = KernelPCA(n_components=n_components, kernel='linear')
        if reducer is 'spca':
            redu = SparsePCA(n_components=n_components, alpha=1, ridge_alpha=0.01,
                             max_iter=1000, method='lars')
        if reducer is 'lda':
            redu = TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=5)
        red = redu.fit_transform(toreduce) if retrain else redu.fit(toreduce)
        self.train_red = red[:inp.shape[0]]
        self.pred_red = red[inp.shape[0]:]

    def train_classifier(self, inp=None, labels=None, classifier='knn',
                         test='CV', scoring='accuracy', percentage=0.3, proba=True, iter=100):
        """
        will use a classification algorithm and train it on the training
        set using the labels and predict its accuracy

        you can pass it your own inputs and labels (be carefull to reduce their features before hand
        or use the ones that are stored in the file

        Params:
        ------
        inp: np.array[values,features], the input array you will train on
        labels: list of values, the input array you have and want to reduce and predict
        classifier: str, the classification algorithm to use (adaboost *, knn *****, )
        test: str, the test algorithm to use (reg,CV)
        scoring: string, the scoring to use (not all of them work for this type of classification)
        percentage: float, the percentage of your data that should be used for testing
        for the regular testing algorithm
        proba: flag, to say if you want the algorithm to compute the probability of each class
        (uniquely for the svm)
        iter: int, number of iterations for the gradient descent of the gaussian mixture classifier

        Returns:
        ------
        score, float, the final score the classifier had

        Outs
        ----

        clf: will save the classifier
        classifier: and its name
        """
        if inp is None:
            inp = self.train_red
        if labels is None:
            labels = self.labeled['ancestry']
        self.classifier = classifier
        if classifier is 'adaboost':
            self.clf = AdaBoostClassifier(n_estimators=int(self.nbcomp * 0.7))
        elif classifier is 'knn':
            self.clf = NearestCentroid()
        elif classifier is 'svm':
            self.clf = SVC(C=1.0, kernel='rbf', degree=3, gamma='auto',
                           coef0=0.0, shrinking=True,
                           probability=proba, tol=0.001, cache_size=400,
                           class_weight=None, verbose=False, max_iter=-1)
        elif classifier is 'gaussian':
            self.clf = GCP(max_iter_predict=iter)
        else:
            print "unkown classifier"
        if test is 'CV':
            scores = cross_val_score(self.clf, inp, labels, scoring=scoring, cv=3, n_jobs=1)
            print "cv scores : " + str(scores)
            score = np.mean(scores)
        elif test is 'reg':
            X_train, X_test, y_train, y_test = self.get_training_data(inp, percentage=0.3)
            self.clf.fit(X_train, y_train)
            y_pred = self.clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
        self.clf.fit(inp, labels)
        print "the total score is of " + str(score)
        return score

    def predict_labels(self, inp=None):
        """
        give it an input (that you have been passed already to the PCA algorithm precising
        that it has already been trained) and gives you the labels

        Params:
        ------
        inp: np.array[values,features], the input array you will train on ( optional)

        Returns:
        -------
        found : list, of found values (saved in the class)
        """
        if self.clf is not None:
            self.found = self.clf.predict(inp) if inp is not None else self.clf.predict(
                self.pred_red)
            return self.found

    def compute_features_nb(self, classifier='knn', vmin=50, vmax=1000, step=10):
        """
        computes the number of features that is the best with a simple gready search
        does not count as training

        Params:
        ------
        classifier: string : name of the classifier for which you want to the best number of
        features
        vmin : int minimal value
        vmax :
        step :

        Returns:
        -------
        a plt plot
        scores : np.array the ordered list of best scores
        vals: list the corresponding ordered values

        """
        vals = range(vmin, vmax, step)
        scores = np.zeros(len(vals))
        for i, val in enumerate(vals):
            self.reducedim(n_components=val)
            score = self.train_classifier(classifier=classifier)
            scores[i] = score
        plt.plot(scores, vals)
        ind = np.argsort(scores)
        scores[:] = scores[ind]
        vals = [vals[i] for i in ind]
        return scores, vals


def _inpar(chrom, label_names, test_names):
    """
    a private function that is called by par_load_from_vcf and should not be used by user
    to understand this function please see load_from_vcf
    """
    rec = {}
    dtrain_gt = []
    dtrain_pha = []
    dtest_gt = []
    dtest_pha = []
    vcf_reader = vcf.Reader(open("data/chunks/inpar_ch" + chrom + ".recode.vcf", 'r'))
    for i, record in enumerate(vcf_reader):
        train_gt = []
        train_pha = []
        test_gt = []
        test_pha = []
        if i % 500 == 0:
            print "done " + str(i) + " of chrom " + str(chrom)
        if record.CHROM != chrom:
            raise ValueError("the file has another type of chromosomes")
        rec.update({record.POS: [record.REF, record.ALT]})
        for name in label_names:
            train_gt.append(record.genotype(
                name).gt_type if record.genotype(name).gt_type is not None else 0)
            train_pha.append(record.genotype(
                name).phased if record.genotype(name).phased is not None else 0)
        for name in test_names:
            test_gt.append(record.genotype(
                name).gt_type if record.genotype(name).gt_type is not None else 0)
            test_pha.append(record.genotype(
                name).phased if record.genotype(name).phased is not None else 0)
        dtrain_gt.append(train_gt)
        dtrain_pha.append(train_pha)
        dtest_gt.append(test_gt)
        dtest_pha.append(test_pha)
    print "finished chrom " + str(chrom)
    return [rec, dtrain_gt, dtrain_pha, dtest_gt, dtest_pha]


def convertlist(longlist, type=None):
    """
    it makes the conversion of list of lists to np array, faster than np.array
    enter list of list
    ouput array of type, type
    """
    tmp = list(chain.from_iterable(longlist))
    return np.array(tmp, dtype=type).reshape((len(longlist), len(longlist[0])))
