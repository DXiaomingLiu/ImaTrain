import glob
import allel
import numpy as np
from keras.preprocessing import sequence
from ImaGene_Xiao import *
from sklearn.neighbors import NearestNeighbors
class ImaVcf:
    '''
    This class takes a directory and reads all of its vcf files. Genotype (haplotype) arrays
    are reshaped into chromosome * variable site * channel. Genotype arrays can be converted
    stored either in simple numpy array format or ImaGene objects.
    Coloured genotype array takes the typ of mutation into account and makes 3-channel (RBG)
    images.
    '''
    def __init__(self,vcf_dir):
        '''

        :param vcf_dir: vcf directory path (remember to include '/' at the end
        '''
        self.vcf_dir = vcf_dir
        self.vcfs = glob.glob("{}./*.vcf".format(self.vcf_dir))
        self.vcf_num = len(self.vcfs)
        self.files = [None] * self.vcf_num
        self.gts = [None] * self.vcf_num
        self.f_gts = [None] *self.vcf_num
        self.colour_snps = [None] * self.vcf_num
        self.pos = [None] * self.vcf_num
        self.labels = [None] * self.vcf_num
        self.label = ['Unknown']*self.vcf_num
        self.headers = []
        self.imagenes = []
        for i in range(self.vcf_num):
            self.headers.append(allel.read_vcf_headers(self.vcfs[i]))
            file = allel.read_vcf(self.vcfs[i])
            self.files[i] = file
            '''Reading, converting and storing the genotypes of polymorphic sites'''
            gt = allel.GenotypeArray(file['calldata/GT'])
            pos = file['variants/POS']
            gt_dim = gt.shape  # [variable site][chromosomes][channel*2]
            count = gt.count_alleles(max_allele=1)
            filtered_gt = []
            filtered_pos = []
            '''
            For each entry in the count matrix (SNP X Chromosomes), if the alternative
            allele frequency is greater than zero in the population (count[row][1]=0), 
            the SNP belongs to the population is selected into the filtered genotype array.
            '''
            for row in range(len(count)):
                if count[row][1]>0:
                    filtered_gt.append(gt[row])
                    filtered_pos.append(pos[row])
            gt = np.asarray(filtered_gt)

            gt_dim = gt.shape
            new_dim = (gt_dim[0],gt_dim[1]*gt_dim[2]) #dim[2] = 2
            gt = gt.reshape(new_dim)
            gt = np.transpose(gt)
            gt = gt.reshape(list(new_dim).append(1)) #adds the channel after

            #count = gt.count_alleles(max_allele=1)
            '''
            filered_gt = np.zeros(198,)
            for j in range(len(count)):
                if count[i][1] > 0:
                    for i in range(198):
                        filered_gt[i].append(gt[i][j])
            '''
            self.gts[i] = gt
            self.pos[i] = filtered_pos

            '''Reading, converting and storing positions of polymorphic sites'''

        self.allele_to_color = {'A': (255 / 2, 255 / 2, 0), 'C': (0, 255 / 2, 255 / 2), 'G': (255 / 2, 0, 255 / 2),
                               'T': (255 / 2, 255 / 2, 255 / 2)}
        self.indel = (255, 255, 255)
        self.to_relative()
    def to_relative(self):
        '''
        :param pos: list of raw positions; dtype: list
        :param center: center loci position; dtype: integer
        :return: list of relative positions; dtype: list
        '''
        for index1,p in enumerate(self.pos):
            left = min(p)
            length = max(p)-left
            for index2, num in enumerate(p):
                p[index2]= (num-left)/length
            self.pos[index1] = p
        return self.pos
    def resize(self, size=(198,400),mode='pad'):
        chrom_num = size[0]
        snp_num = size[1]
        if mode=='pad':
            for index,image in enumerate(self.gts):
                dim = image.shape
                image = image.reshape((dim[0],dim[1])) if len(dim)>3 else image
                self.gts[index]=sequence.pad_sequences(image,padding='post',  maxlen=snp_num)
            self.pos = sequence.pad_sequences(self.pos, padding='post', value=-1,
                                                     dtype='float32', maxlen=snp_num)
        elif mode=='trans':
            pass
    def sort(self):
        for i,image in enumerate(self.gts):
            self.gts[i]=self.sort_min_diff(image)

    def sort_min_diff(self, amat):
        '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
        this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
        assumes your input matrix is a numpy array (Schrider)'''
        mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
        v = mb.kneighbors(amat)
        smallest = np.argmin(v[0].sum(axis=1))
        return amat[v[1][smallest]]
    def colour_map_iterator(self,snp):
        return self.colour_scheme[snp]
    def to_coloured(self):
        return np.array(map(self.colour_map_iterator(),))
    def get_gts(self):
        '''
        :return: list containing arrays of genotypes
        '''
        return self.gts
    def get_pos(self):
        return self.pos
    def to_ImaGene(self):
        '''
        You need my modified version of ImaGene for this to work - currently named ImaGene_Xiao.py (see my GitHub),
        because Matteo's original ImaGene was only intended for msms simulation files
        :return: ImaGene objects
        '''
        for i in range(self.vcf_num):
            self.imagenes[i] = ImaGene(data=self.gts[i], positions=self.pos[i],
                                       description='A Vcf ImaGene: {0}'.format(self.vcfs[i]),
                                       targets=self.labels[i], parameter_name=None, classes=[])
    def get_ImaGene(self):
        return self.imagenes
    def to_coloured(self):
        '''
        This doesn't quite work well now - it's slow and the colouring scheme is not optimal; work in progress
        '''

        '''
        for i in range(self.vcf_num):
            file = self.files[i]
            alt = file['variants/ALT']
            ref = file['variants/REF']
            # print(dim,alt.shape,ref.shape)
            # print(file['variants/numalt'])
            # print(alt)
            
            colour_gt = np.zeros(dim, dtype=np.dtype((np.float32, (1, 3))))
            for y in range(dim[0]):  # for each snp
                for x in range(dim[1]):  # in each individual
                    for allele in range(2):  # for each chromosome
                        gt_index = gt[y][x][allele]
                        if gt_index:  # 0,1,2... corresponding to ref and alternative alleles in that order
                            nt = alt[y][
                                gt_index - 1]  # -1 because starting with the first alternative allele (1-1=0 - first alt allele)
                        else:
                            nt = ref[y]
                        channel = self.allele_to_color[nt] if len(nt) == 1 else self.indel
                        # print(channel)
                        # print(type(channel))
                        colour_gt[y][x][allele] = channel
        '''
        pass
    def set_colour_scheme(self,allele_scheme, indel_scheme = (255, 255, 255)):
        '''Input: dictionary with keys: A, C, G and T'''
        self.allele_to_colour = allele_scheme
        self.indel = indel_scheme
    def set_label(self,name, label):
        '''
        :param name: full name of the vcf file to be labeled (with .vcf); dtype: string
        :param label: manually designated label; dtype: string
        :return:
        '''
        index = self.vcfs.index(name)
        self.labels[index] = label

    def save_images(self,file,index):
        data = self.gts[index]
        plt.imshow(data, cmap='Greys', interpolation='nearest')
        plt.savefig('{0}_{1}.png'.format(file, index))
        pass
    def save_npy(self,dest):
        np.save('{0}test_GT.npy'.format(dest), np.asarray(self.gts))
        #np.save('{0}test_coloured_GT.npy'.format(dest), self.colour_snps)
        np.save('{0}test_POS.npy'.format(dest), np.asarray(self.pos))  # maybe use difference at the end?
        np.save('{0}test_LABEL.npy'.format(dest), np.asarray(self.labels))