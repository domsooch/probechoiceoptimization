from __future__ import print_function
import sys, os, random, math, copy

# testBit() returns a nonzero result, 2**offset, if the bit at 'offset' is one.

def testBit(int_type, offset):
    mask = 1 << offset
    return(int_type & mask)

# setBit() returns an integer with the bit at 'offset' set to 1.

def setBit(int_type, offset):
    mask = 1 << offset
    return(int_type | mask)

# clearBit() returns an integer with the bit at 'offset' cleared.
def clearBit(int_type, offset):
    mask = ~(1 << offset)
    return(int_type & mask)

# toggleBit() returns an integer with the bit at 'offset' inverted, 0 -> 1 and 1 -> 0.

def toggleBit(int_type, offset):
    mask = 1 << offset
    return(int_type ^ mask)
def str_reverse(s):
    s = list(s)
    s.reverse()
    return ''.join(s)

DEBUG=True
class BitArray:
    def __init__(self, size, label='', data_packet=[]):
        self.label = label
        self.data_packet = data_packet
        self.max = size
        self.array_sz = int(size/64)+1
        mask = 0# << 64
        self.array = [mask for i in range(self.array_sz)]
        self.bf_size=self.array_sz*64
        self.debugLst = []
    def __getitem__(self, offset):
        return self.array[int(offset/64)] & (1 << offset%64)
    def __setitem__(self, offset, m):
        if DEBUG:
            if not(offset in self.debugLst):
                if m:
                    self.debugLst.append(offset)
                    self.debugLst.sort()
                else:
                    if offset in self.debugLst:
                        del self.debugLst[self.debugLst.index(offset)]
        if m:
            self.array[int(offset/64)] = self.array[int(offset/64)] | (1 << offset%64)
        else:
            self.array[int(offset/64)] = self.array[int(offset/64)] & ~(1 << offset%64)
    def count(self):
        count = 0
        for idx in range(self.array_sz):
            int_type = self.array[idx]
            while(int_type):
                count += (int_type & 1)
                int_type >>= 1
        return count
    def debug_count(self):
        count = 0
        for idx in range(self.array_sz):
            int_type = self.array[idx]
            while(int_type):
                count += (int_type & 1)
                int_type >>= 1
        if DEBUG:
            assert(len(self.debugLst)==count)
        return count
    def clearBits(self, offsetOrLst):
        if type(offsetOrLst)==type(0):
            offsetOrLst=[offsetOrLst]
        for offset in offsetOrLst:
            if DEBUG:
                if offset in self.debugLst:
                    del self.debugLst[self.debugLst.index(offset)]
            self.array[int(offset/64)] = self.array[int(offset/64)] & ~(1 << offset%64)
    def setBits(self, offsetOrLst):
        if type(offsetOrLst)==type(0):
            offsetOrLst=[offsetOrLst]
        for offset in offsetOrLst:
            self[offset]=1
    def __and__(self, other):
        assert (len(other.array) == len(self.array))
        o = self.cleanCopy()
        for a in range(len(self.array)):
            o.array[a] = (self.array[a])&(other.array[a])
        return o
    def __or__(self, other):
        assert (len(other.array) == len(self.array))
        o = self.cleanCopy()
        for a in range(len(self.array)):
            o.array[a] = (self.array[a])|(other.array[a])
        return o
    def __xor__(self, other):
        assert (len(other.array) == len(self.array))
        o = self.cleanCopy()
        for a in range(len(other.array)):
            o.array[a] = (self.array[a])^(other.array[a])
        return o
    def cleanCopy(self):
        """Makes an empty copy or self"""
        return BitArray(self.max, label=self.label, data_packet=self.data_packet)
    def copy(self):
        """Makes an exact copy or self"""
        b = BitArray(self.max, label=self.label, data_packet=self.data_packet)
        b.array=copy.deepcopy(self.array)
        b.debugLst=copy.deepcopy(self.debugLst)
        return b
    def setRange(self, start, end):
        for i in range(min(start, end), max(start, end), 1):
            self[i]=1
    def ANDFuse(self, other):
        #converts self to outcome of AND
        assert (len(other.array) == len(self.array))
        for a in range(len(other.array)):
            self.array[a] = (self.array[a])&(other.array[a])
    def ORFuse(self, other):
        #converts self to outcome of OR
        assert (len(other.array) == len(self.array))
        for a in range(len(other.array)):
            self.array[a] = (self.array[a])|(other.array[a])
    def XORFuse(self, other):
        #converts self to outcome of XOR
        assert (len(other.array) == len(self.array))
        for a in range(len(other.array)):
            self.array[a] = self.array[a]^other.array[a]
    def testBit(self, offset):
        idx, mask = self.find_bit(offset)
        return self.array[idx] & mask
    def find_bit(self, offset, m=1):
        idx = int(offset/64)
        mask = m << offset%64
        return idx, mask
    def exportfp(self, fp):
        oLst = [self.label, self.max] +['\t'.join([str(d) for d in self.data_packet])] +  self.array
        buff = '\n'.join([str(x) for x in oLst])
        fp.write(buff)
    def exportfn(self, ofn):
        fp = open(ofn, 'w')
        self.exportfp(fp)
        fp.close()
    def importFromRawLst(self, Lst):
        self.array = []
        for s in Lst:
            self.array.append(int(s))
        self.array_sz = len(self.array)
    def importfn(self, ifn):
        buff = open(ifn).read()
        Lst = buff.split('\n')
        self.label = Lst.pop(0)
        s = Lst.pop(0)
        self.max = int(s)
        self.data_packet = Lst.pop(0).split('\t')
        self.array = []
        for s in Lst:
            self.array.append(int(s))
        self.array_sz = len(self.array)
    def export_bit_lst(self):
        return [1 if self[offset] else 0 for offset in range(self.max)]
    def export_bit_idxLst(self):
        l = []
        for offset in range(self.max+1):
            if self.testBit(offset):
                l.append(offset)
        return l
    def __str__(self):
        sLst = []
        sLst.append('%s array_sz: %i count: %i'%(self.label, self.array_sz, self.count()))
        sLst.append(self.bin_str())
        return '\n'.join(sLst)
    def Part(self):
        s = float(self.count())/self.max
        return s
    def bin_str(self):
        sLst = []
        for idx in range(self.array_sz):
            b_str=bin(self.array[idx])[2:]
            b_str='0'*(64-len(b_str))+b_str
            sLst.append(str_reverse(b_str))
        b=''.join(sLst)
        return b
    def test(self):
        bin_str=self.bin_str()
        print('export_bit_idxLst', self.export_bit_idxLst(), '\n',
              'filter(lambda i:self[i]', list(filter(lambda i:self[i]!=0, range(self.max+1))),'\n',
              'self.debugLst', self.debugLst, '\n',
              'count', self.debug_count(), '\n',
              'bin_str()', bin_str, '\n',
              'count(bin_str)', sum([int(x) for x in list(bin_str)]), '\n',
              self.array)

#This version uses a single bitfield built from an int, but it is at least 4x slower
#Unless I copy it in place on each cycle, it nust be getting cached somehwere if not copied immediately
class alt_BitArray:
    def __init__(self, size, label='', data_packet=[]):
        self.label = label
        self.data_packet = data_packet
        self.max = size
        self.bf = 1 <<self.max+1
        self.debugLst = []
    def __getitem__(self, offset):
        assert(offset<self.max)
        return self.bf&(1 << offset)
    def __setitem__(self, offset, m):
        assert(offset<self.max)
        if DEBUG:
            if not(offset in self.debugLst):
                if m:
                    self.debugLst.append(offset)
                    self.debugLst.sort()
                else:
                    if offset in self.debugLst:
                        del self.debugLst[self.debugLst.index(offset)]
        if m:
            self.bf |= (1 << offset)
        else:
            self.bf &= ~(1 << offset)
    def count(self):
        count = 0
        for idx in range(self.max):
            if self.bf&(1 << idx):count+=1
        return count
    def debug_count(self):
        count = 0
        for idx in range(self.max):
            if self.bf&(1 << idx):count+=1
        if DEBUG:
            assert(len(self.debugLst)==count)
        return count
    def clearBits(self, offsetOrLst):
        if type(offsetOrLst)==type(0):
            offsetOrLst=[offsetOrLst]
        for offset in offsetOrLst:
            if DEBUG:
                if offset in self.debugLst:
                    del self.debugLst[self.debugLst.index(offset)]
            self[offset]=0
    def setBits(self, offsetOrLst):
        if type(offsetOrLst)==type(0):
            offsetOrLst=[offsetOrLst]
        for offset in offsetOrLst:
            self[offset]=1
    def __and__(self, other):
        assert (other.max == self.max)
        o = self.cleanCopy()
        o.bf = self.bf&other.bf
        return o
    def __or__(self, other):
        assert (other.max == self.max)
        o = self.cleanCopy()
        o.bf = self.bf|other.bf
        return o
    def __xor__(self, other):
        assert (other.max == self.max)
        o = self.cleanCopy()
        o.bf = self.bf^other.bf
        return o
    def cleanCopy(self):
        """Makes an empty copy or self"""
        return BitArray(self.max, label=self.label, data_packet=self.data_packet)
    def copy(self):
        """Makes an exact copy or self"""
        b = BitArray(self.max, label=self.label, data_packet=self.data_packet)
        b.array=copy.deepcopy(self.bf)
        b.debugLst=copy.deepcopy(self.debugLst)
        return b
    def setRange(self, start, end):
        for i in range(min(start, end), max(start, end), 1):
            self[i]=1
    def ANDFuse(self, other):
        #converts self to outcome of AND
        assert (other.max == self.max)
        self.bf=self.bf&other.bf
    def ORFuse(self, other):
        #converts self to outcome of OR
        assert (other.max == self.max)
        self.bf=self.bf|other.bf
    def XORFuse(self, other):
        #converts self to outcome of XOR
        assert (other.max == self.max)
        self.bf=self.bf^other.bf
    def export_bit_lst(self):
        return [1 if self[offset] else 0 for offset in range(self.max)]
    def export_bit_idxLst(self):
        l = []
        for offset in range(self.max):
            if self[offset]:
                l.append(offset)
        return l
    def __str__(self):
        sLst = []
        sLst.append('%s max_sz: %i count: %i'%(self.label, self.max, self.count()))
        sLst.append(self.bin_str())
        return '\n'.join(sLst)
    def Part(self):
        s = float(self.count())/self.max
        return s
    def bin_str(self):
        sLst = []
        b_str=bin(self.bf)[3:]
        return str_reverse(b_str)
    def test(self):
        bin_str=self.bin_str()
        print('export_bit_idxLst', self.export_bit_idxLst(), '\n',
              'filter(lambda i:self[i]', list(filter(lambda i:self[i]!=0, range(self.max))),'\n',
              'self.debugLst', self.debugLst, '\n',
              'count', self.debug_count(), '\n',
              'bin_str()', bin_str, '\n',
              'count(bin_str)', sum([int(x) for x in list(bin_str)]))

# class Genome_BitArray(BitArray, object):
#     def __init__(self, fa_tiling_obj):
#         self.fa_tiling_obj = fa_tiling_obj
#         x = super(Genome_BitArray, self)
#         x.__init__(fa_tiling_obj.seq_len, label=fa_tiling_obj.label, data_packet=[])
#     def export(self):
#         pass
#ToDo Test This
class Genome_BitArray(BitArray):
    """
    Used for masking genome seqquences and then selecting primers from
    alloweable regions
    """
    def __init__(self, fa_tiling_obj=None):
        print('Genome_BitArray is untested')
        if fa_tiling_obj:
            self.fa_tiling_obj = fa_tiling_obj
            self.nonn_seq_len = self.fa_tiling_obj.nonn_seq_len
            BitArray.__init__(self, fa_tiling_obj.seq_len, label=fa_tiling_obj.label, data_packet=['nonn_seq_len', self.nonn_seq_len, 'fa_tiling_obj.label', self.fa_tiling_obj.label])
        else:
            self.fa_tiling_obj =None
            self.nonn_seq_len = -1
            BitArray.__init__(self, 0, '', data_packet=[])
        self.num_N_bases = -1
    def cleanCopy(self):
        return Genome_BitArray(self.fa_tiling_obj)
    def exactCopy(self):
        c = Genome_BitArray(self.fa_tiling_obj)
        c.ORFuse(self)
        return c
    def bitCopy(self):
        #Does not maintain the dna sequence association
        c = self.cleanCopy()
        c.nonn_seq_len = self.nonn_seq_len
        c.num_N_bases = self.num_N_bases
        return c
    def non_n_Part(self):
        return float(self.count())/self.nonn_seq_len
    def maskN(self):
        if self.fa_tiling_obj:
            print ('maskN: ',self.label, " num_N_bases:",)
            seq = self.fa_tiling_obj.GetSeq()
            for b in range(len(seq)):
                if seq[b] == 'N':
                    self[b]=1
            self.num_N_bases = self.count()
            print (self.num_N_bases, 'of', self.max, 'self.nonn_seq_len', self.nonn_seq_len)
    def mask_seq(self, inseq, non_base='.', BitOnMeansPassThrough=True):        
        if BitOnMeansPassThrough:
            olst=[non_base for i in range(len(inseq))]
            for idx in self.export_bit_idxLst(self):
                olst[idx]=inseq[idx]
        else:
            olst=list(inseq)
            for idx in self.export_bit_idxLst(self):
                olst[idx]='N'
        return ''.join(olst)
    def export_masked_seq(self, non_base='N', BitOnMeansPassThrough=True, Part_min=0.05):
        part = self.Part()
        if BitOnMeansPassThrough:
            if part <Part_min:
                print ("No sequence made it for %s Part: %f"%(self.fa_tiling_obj.label, part))
        maskType = 'Mask-ON' if (BitOnMeansPassThrough==True) else 'Mask-Off'
        label = "%s|%s-masked_part:%.2f|"%(self.fa_tiling_obj.label, maskType, part)
        label = label.replace('||', '|')
        oseq = self.mask_seq(self.fa_tiling_obj.GetSeq(), non_base=non_base, BitOnMeansPassThrough=BitOnMeansPassThrough)
        return [label, oseq, len(oseq)]
    def stats(self):
        return [self.max, self.num_N_bases, self.nonn_seq_len, self.Sum()]
    def save_file(self, save_dir, subj_label, query_idx, subj_idx):
        self.data_packet.extend(['subj_label', subj_label, 'query_idx', query_idx, 'subj_idx', subj_idx])
        ofn = 'q_%i__s%i.txt'%(query_idx, subj_idx)
        opath = os.path.join(save_dir, ofn)
        self.exportfn(opath)
        print ('saved ', opath)
     
        
        
        
if __name__ == "__main__":
    pass
    
    
    
    
    
