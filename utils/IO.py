print('IO.py')
import sys
import os
import fileinput, csv, fnmatch




def unzip(p):
    cmd = "gzip -d %s"%(p)
    print(cmd)
    os.system(cmd)
    return p.replace('.gz','')

def CSVimport(csv_path):
    with open(csv_path, 'rb') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(9024))
        print(str(dialect))
        #raw_input('key')
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        for rLst in reader:
            #if not(type(rLst)==type([])): rLst = rLst.split(',')
            yield rLst

def simple_inCSV(csv_path):
    b = open(csv_path, 'r').read()
    lineLst = b.split('\n')
    print('simple_inCSV %s  chars: %i lines %i' % (csv_path, len(b), len(lineLst)))
    return [l.split(',') for l in lineLst]


def inCSV(csv_path, LL_Ignore = False):
    c = CSVimport(csv_path)
    olst=[]
    for rec in c:
        if LL_Ignore:
            LL_Ignore = False
            continue
        olst.append(rec)
    return olst

def inTSV(inp, LL_Ignore = False):
    buff = open(inp,'r').read()
    olst=[]
    for rec in buff.split('\n'):
        rec=rec.split('\t')
        if LL_Ignore:
            LL_Ignore = False
            continue
        olst.append(rec)
    return olst

def CSV_Export(lst, ofn, Verbose=False):
    if not(lst):return
    oLst = []
    for r in lst:
        oLst.append([str(e) for e in r])
    WriteLstToFile(oLst, ofn, delim=',', Verbose=Verbose)
    return ofn

def inFASTA_split(inp):
    buff = open(inp,'r').read()
    fLst = buff.split('>')
    oLst = []
    for f in fLst:
        lst = f.split('\n')
        label = lst.pop(0)
        seq = ''.join(lst)
        seq_len = len(seq)
        oLst.append([label, seq, seq_len])
    return oLst

def inFASTA_yield(inpORLst):
    num_seqs = 0
    if type(inpORLst) is type(''):
        inpLst = [inpORLst]
    else:
        inpLst = inpORLst
    for inp in inpLst:
        seqLst = []
        label=''
        line = True
        with open(inp, 'r') as f:
            c=0
            while 1:
                line = f.readline()
                if not(line):
                    c+=1
                    if c>5:break
                line = line.replace('\n', '').replace('\r', '')
                if ('>' in line[:2]):
                    if label and seqLst:
                        num_seqs+=1
                        seq = ''.join(seqLst)
                        seqLst = []
                        seq_len = len(seq)
                        yield [label[1:].strip(), seq, seq_len]
                    label = line
                else:
                    seqLst.append(line.strip())
        if label and seqLst:
            num_seqs+=1
            seq = ''.join(seqLst)
            seqLst = []
            seq_len = len(seq)
            yield [label[1:], seq, seq_len]
    print("inFASTA_yield(%s)  num_seqs: %i"%(inp, num_seqs))

def inFASTA(inp):
    lst = []
    for fa in inFASTA_yield(inp):
        lst.append(fa)
    return lst

# def inFASTA(inp):
#     buff = open(inp,'r').read()
#     fLst = buff.split('\n')
#     oLst = []
#     l = 0
#     while l < len(fLst):
#         line = fLst[l]
#         if ('>' in line[:2]):
#             label = line
#             seqLst = []
#             while l < len(fLst)-1:
#                 l+=1
#                 line = fLst[l]
#                 if '>' in line[:3]:
#                     seq = ''.join(seqLst)
#                     seq_len = len(seq)
#                     oLst.append([label, seq, seq_len])
#                     label = line
#                     break
#                 else:
#                     seqLst.append(line)
#         l+=1
#     if seqLst:
#         seq = ''.join(seqLst)
#         seq_len = len(seq)
#         oLst.append([label, seq, seq_len])
#     return oLst
#
# def inFASTA_yield(inp):
#     seqLst = []
#     label=''
#     num_seqs = 0
#     if type(inp)==type([]):
#         inLst = inp
#     else:
#         inLst=[inp]
#     FI = fileinput.input(inLst)
#     for line in FI:
#         line = line.replace('\n', '').replace('\r', '')
#         if ('>' in line[:2]):
#             if label and seqLst:
#                 num_seqs+=1
#                 seq = ''.join(seqLst)
#                 seqLst = []
#                 seq_len = len(seq)
#                 yield [label[1:], seq, seq_len]
#             label = line
#         else:
#             seqLst.append(line.strip())
#     if label and seqLst:
#         num_seqs+=1
#         seq = ''.join(seqLst)
#         seqLst = []
#         seq_len = len(seq)
#         yield [label[1:], seq, seq_len]
#     print("inFASTA_yield(%s)  num_seqs: %i"%(inp, num_seqs))


def clean_fasta_label(label, max_len=50):
    """fasta labels cannot have spaces and cannot be longer than about 200 chars, and can't have formatting crap in them"""
    label = label.strip()
    bad_chars = ['>', ',', '|','\n','\t','[', ']']
    for b in bad_chars:
        label = label.replace(b,'')
    while '  ' in label:
        label = label.replace('  ', ' ')
    label = label.replace(' ', '_')
    return label[:max_len]


def clean_fasta_sequence(seq, splitN=None):
    seq = seq.strip()
    bad_chars = ['>', ',', '|','\n', '\r',' ', '\t','0','1','2','3','4','5','6','7','8','9']
    for b in bad_chars:
        seq = seq.replace(b,'')
    if splitN == None: return seq
    return '\n'.join([seq[i:i+splitN] for i in range(0, len(seq), splitN)])


def concat_fasta_records(fasta_path, fasta_label, out_fp):
    out_fp.write('>%s\n'%fasta_label)
    for fasta_rec in IO.inFASTA_yield(fasta_path):
        label, seq = fasta_rec[:2]
        if not(seq):
            print("WARNING: MIssing One Sequence: %s"%label)
            continue
        out_fp.write('%s\n%s\n'%(IO.clean_fasta_sequence(seq, splitN=1000), 'N'*40))


def combine_fasta_records(out_fasta_path, inpathLst, splitN=None):
    with open(out_fasta_path, 'w') as out_fp:
        for fasta_path in inpathLst:
            for fasta_rec in inFASTA(fasta_path):
                label, seq = fasta_rec[:2]
                out_fp.write('>%s\n%s\n'%(label, IO.clean_fasta_sequence(seq, splitN=splitN)))
    out_fp.close()
    return out_fasta_path

def outFasta(faLst, out_path):
    b = ''
    for fa in faLst:
        l, f = fa[:2]
        b+= '>%s\n%s\n'%(l, f)
    ofp = open(out_path, 'w')
    ofp.write(b)
    ofp.close()

def toSeqDict(faLst):
    d= {}
    for fa in faLst:
        d[fa[0]] = fa[1]
    return d


def inFASTA_dict(BI_ifp):
    return toSeqDict(inFASTA(BI_ifp))



class outFasta_stream:
    def __init__(self, outpath=None, outfp=None, max_cols = 1000, write_buff_sz=1000):
        self.current_genome_label = ''
        if outfp==None and outpath==None:
            raise('You need either of [outfp, outpath]:', outfp, outpath)
        if outfp and outpath:
            raise('You need only ONE of [outfp, outpath]:', outfp, outpath)
        if outfp==None and outpath:
            self.outfp = open(outpath, 'w')
        else:
            self.outfp = outfp
        self.max_cols = max_cols
        self.write_buff_sz=write_buff_sz
        self.buff = []
        
        self.out_fn = os.path.basename(outpath)
        self.genomes_written=0
        self.seqs_written = 0
        self.total_seq_len = 0
        self.base_position = 0
        self.current_genome_seq_len = 0
        self.current_genome_base_position = 0
    def write(self, faLst):
        for fa in faLst:
            label, seq = fa[:2]
            if not(label[0]=='>'): label = '>' + label.strip()
            self.buff.append(label)
            for i in range(0, len(seq), self.max_cols):
                s=seq[i:i+self.max_cols]
                self.total_seq_len +=len(s)
                self.buff.append(s)
            self.seqs_written+=1
            if self.write_buff_sz and len(self.buff)>self.write_buff_sz:
                print('outFasta_stream:write %i recs'%len(self.buff))
                self.outfp.write('\n'.join(self.buff)+'\n')
                self.buff = []
        self.SaveBuffer()
    def write_Nbuffered(self, fa):
        #This is designed for fractured genomes that have many many contigs this pads the interstitial regions with N's
        #Feed one fasta record at a time
        labelLst = []
        label, seq = fa[:2]
        if not(label[0]=='>'): label = '>' + label.strip()
        print('ignoring label: ', label)
        start_point = self.base_position
        for i in range(0, len(seq), self.max_cols):
            s=seq[i:i+self.max_cols]
            len_s = len(s)
            self.total_seq_len +=len_s
            self.base_position +=len_s
            self.buff.append(s)
        self.base_position+= self.NBuffer_Previous()
        self.seqs_written+=1
        labelLst.append([self.out_fn, label, start_point, self.base_position, self.base_position-start_point])
        return labelLst        
    def NBuffer_Previous(self):
        #buffer a fasta file
        if not(self.buff):return
        if self.buff[-1][0] == '>':return
        Nbuffer_sz = self.max_cols - len(self.buff[-1])
        Ns_added = 0
        if Nbuffer_sz>0:
            Ns_added +=Nbuffer_sz
            self.buff.append('N'*Nbuffer_sz)
        if Nbuffer_sz<100:
            Ns_added +=self.max_cols
            self.buff.append('N'*self.max_cols)
        return Ns_added
    def SaveBuffer(self):
        if self.buff:
            self.outfp.write('\n'.join(self.buff)+'\n')
            self.buff = []
    def InsertNextGenomeLabel(self, label):
        self.SaveBuffer()
        if not(label[0]=='>'): label = '>' + label.strip()
        self.buff.append(label)
        if self.current_genome_label:
            previous_genome_data =  [self.out_fn, self.current_genome_label, self.total_seq_len-self.current_genome_seq_len, self.base_position-self.current_genome_base_position]
            self.current_genome_seq_len = self.total_seq_len
            self.current_genome_base_position = self.base_position
            self.current_genome_label = label
        else:
            self.current_genome_label = label
            previous_genome_data =  ['current_genome_label', 'total_seq_len', 'N_base_position']
        return previous_genome_data
    def close(self):
        if not(self.outfp.closed):
            self.SaveBuffer()
            self.outfp.close()
        print('outFasta_stream::close: %s outFasta_stream: wrote %i seqs %i bp'%(self.out_fn, self.seqs_written, self.total_seq_len))
        return [self.out_fn, self.genomes_written, self.seqs_written, self.total_seq_len, self.base_position]
        
        

def WriteLstToFile(outDB, opath, delim = '\t', buff_sz = 1000, Verbose=False):
    if Verbose: print('WriteLstToFile: %s %i recs' %(opath, len(outDB)))
    ostr = ''
    LineLst = []
    c = 0
    ofp = open(opath, 'w')
    for line in outDB:
        c +=1
        line = [str(obj) for obj in line]
        line = delim.join(line)
        LineLst.append(line)
        if c > buff_sz:
            c = 0
            ostr = '\n'.join(LineLst) + '\n'
            ofp.write(ostr)
            ostr = ''
            LineLst = []
    ostr = '\n'.join(LineLst)+'\n'
    ofp.write(ostr)
    ofp.write("\n")
    ofp.close()
    return opath

class Streamwriter:
    def __init__(self, opath, sep='\t', write_sz=1000000):
        self.sep=sep
        self.opath=opath
        self.fn=os.path.basename(self.opath)
        self.n=0
        self.write_sz=write_sz
        self.fp=open(opath, 'w')
        self.lst=[]
    def add(self, rec):
        self.n+=1
        self.lst.append(self.sep.join([str(x) for x in rec]))
        if len(self.lst)>self.write_sz:
            self.write()
    def write(self):
        if self.lst:
            self.fp.write('\n'.join([l for l in self.lst]))
            self.lst=[]
            print ('%s has %i recs at fpos: %ld'%(self.fn, self.n, self.fp.tell()))
    def close(self):
        self.write()
        self.fp.close()
        print ('%s has %i recs fpos: %ld Done'%(self.opath, self.n, os.path.getsize(self.opath)))


