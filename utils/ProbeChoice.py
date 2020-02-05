import sys, os, copy, pprint, glob, random
from importlib import reload
import pathlib, importlib
import numpy as np

from . import BitOperations as B



class BitMap:
    """Stores chosen probes and provides universal indexing for peptides mapping to protein bitfields"""
    def __init__(self):
        self.d={}
        self.n=0
        self.LOCK=False
    def process_peptide(self, peptide):
        for p in peptide.Lst:
            if not(p in self.d):self.d[p]=0
            self.d[p]+=1
    def all_peptide_counts(self):
        return list(self.d.items())
    def order(self, count_limit=None):
        if self.LOCK:
            print('BitMap Has been Locked')
            return None
        lst=list(self.d.items())
        lst.sort(key=lambda x:-x[1])
        self.Lst=[]
        for i in range(len(lst)):
            p,n=lst[i]
            if count_limit and n<count_limit:break
            self.Lst.append(p)
        self.n=len(self.Lst)
        self.LOCK=True
    def write_token_to_bitfield(self, p, bitfield):
        #bitfield[self.Lst.index(p)]
        if not(p in self.Lst):return -1
        try:
            x= self.Lst.index(p)
            bitfield[self.Lst.index(p)]=1
        except:
            print('BFOverLimit:self.Lst.index(%s) bitfield sz: %i, idx: %i'%(p, bitfield.max, self.Lst.index(p)))
            return self.n
    def display(self, limit=10):
        print('BitMap: %i '%len(self.Lst))
        for i in range(min(limit, self.n)):
            p=self.Lst[i]
            print('%s\t%i'%(p, self.d[p]))
            
class Peptide:
    def __init__(self, seq, label, nmer_sz=6):
        self.nmer_sz=nmer_sz
        self.seq=seq
        self.label=label
        self.acc=label.split('|')[1]
        self.Lst=[]
        self.bf=None
        self.nmer_tile()
    def nmer_tile(self, nmer_sz=None):
        nmer_sz= self.nmer_sz if (nmer_sz==None) else nmer_sz
        for i in range(0,len(self.seq)-nmer_sz,1):
            self.Lst.append(self.seq[i:i+nmer_sz])
        self.Lst.sort()
    def make_bitfield(self, bit_map):
        self.bf=B.BitArray(bit_map.n+1)
        for p in self.Lst:
            bit_map.write_token_to_bitfield(p, self.bf)
    def display(self):
        print('acc: %s seq_len: %i, bf_occupancy: %i'%(self.acc, len(self.seq), self.bf.count()))
    
    
import multiprocessing as mp
from contextlib import closing
 
def chooseMostPrevalentProbe(probeLst, probe_bitmap, randomize=True):
    number_Lst=[probe_bitmap.d[probe] for probe in probeLst]
    if randomize:
        n_sum=sum(number_Lst)
        r=int(random.random()*n_sum)
        for i in range(len(number_Lst)):
            r-=number_Lst[i]
            if r<0:break
        return probeLst[i]
    else:
        return probeLst[np.argmax(number_Lst)]
    
    
def GenerateProbeSetBF(probeLst, probe_bitmap):
    probeset_bf=B.BitArray(probe_bitmap.n+1)
    for probe in probeLst:
        probe_idx=probe_bitmap.Lst.index(probe)
        probeset_bf[probe_idx]=1
    return probeset_bf

def ComputeProteinClusters(proteinLst, probeset_bf):
    """This function computes the number of clusters 
    given a set of proteins and choice of probes"""
    proteinprobehit_bfLst=[]
    dedup_dict={}
    for protein in proteinLst:
        pap=protein.bf&probeset_bf
        bstr=pap.bin_str()
        if not bstr in dedup_dict:dedup_dict[bstr]=0
        dedup_dict[bstr]+=1
        proteinprobehit_bfLst.append(pap)
    return len(dedup_dict)

    
def ComputeProteinClusters_DoesNotWork(proteinLst, probeset_bf):
    #ToDo: find efficient/correct way to sort bitfields 
    """This function computes the number of clusters 
    given a set of proteins and choice of probes"""
    proteinprobehit_bfLst=[]
    for protein in proteinLst:
        pap=protein.bf&probeset_bf
        proteinprobehit_bfLst.append(pap)
    #dedupe rec count
    proteinprobehit_bfLst.sort()
    current_rec=proteinprobehit_bfLst[0]
    cluster_count=1
    for rec in proteinprobehit_bfLst:
        if rec!=current_rec:
            cluster_count+=1
            current_rec=rec
    return cluster_count

def ChoiceCycle(proteinLst, probe_bitmap, probeLst, choice_window=0):
    #Computes info gain for each remaining probe
    probeset_bf=GenerateProbeSetBF(probeLst, probe_bitmap)
    probe_choiceLst=[]
    for probe in probe_bitmap.Lst:
        if (probe in probeLst):continue
        probe_idx=probe_bitmap.Lst.index(probe)
        if probeset_bf[probe_idx]:continue
        probeset_bf[probe_idx]=1
        cluster_count, proteinprobehit_bfLst= ComputeProteinClusters(proteinLst, probeset_bf)
        probe_choiceLst.append([cluster_count, probe])
        probeset_bf[probe_idx]=0
    probe_choiceLst.sort(key=lambda x:-x[0])
    top_num_clusters=probe_choiceLst[0][0]
    choiceLst=[]
    for i in range(len(probe_choiceLst)):
        c=probe_choiceLst[i]
        if top_num_clusters-c[0]>=choice_window:
            choiceLst.append(c)
        else:
            break
    print('ChoiceCycle:probe_choiceLst:%i choiceLst:%i top_num_clusters:%i'%(len(probe_choiceLst),len(choiceLst), top_num_clusters))
    return choiceLst

def compute_protein_clusters_Runner(t):
    #For multiprocessing
    proteinLst, probeset_bf, probe_idx, probe = t
    probeset_bf[probe_idx]=1
    cluster_count= ComputeProteinClusters(proteinLst, probeset_bf)
    return [cluster_count, probe]
    
def ChoiceCycle_mp(proteinLst, probe_bitmap, probeLst, choice_window=0, num_ranks=2):
    #Computes info gain for each remaining probe
    #Multiprocessing version of ChoiceCycle
    probeset_bf=GenerateProbeSetBF(probeLst, probe_bitmap)
    probe_choiceLst=[]
    processingLst=[]
    for probe in probe_bitmap.Lst:
        if (probe in probeLst):continue
        probe_idx=probe_bitmap.Lst.index(probe)
        if probeset_bf[probe_idx]:continue
        processingLst.append([proteinLst, probeset_bf.copy(), probe_idx, probe])
    with closing( mp.Pool(num_ranks) ) as p:
        probe_choiceLst=p.map(compute_protein_clusters_Runner, processingLst)
    probe_choiceLst.sort(key=lambda x:-x[0])
    top_num_clusters=probe_choiceLst[0][0]
    choiceLst=[]
    for i in range(len(probe_choiceLst)):
        c=probe_choiceLst[i]
        if top_num_clusters-c[0]>=choice_window:
            choiceLst.append(c)
        else:
            break
    print('ChoiceCycle:probe_choiceLst:%i choiceLst:%i top_num_clusters:%i'%(len(probe_choiceLst),len(choiceLst), top_num_clusters))
    #print(choiceLst)
    return choiceLst

