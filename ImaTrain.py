'''
//                        _oo0oo_
//                      o8888888o
//                      88" . "88
//                      (| -_- |)
//                      0\  =  /0
//                    ___/`---'\___
//                  .' \\|     |// '.
//                 / \\|||  :  |||// \
//                / _||||| -:- |||||- \
//               |   | \\\  -  /// |   |
//               | \_|  ''\---/''  |_/ |
//               \  .-\__  '-'  ___/-. /
//             ___'. .'  /--.--\  `. .'___
//          ."" '<  `.___\_<|>_/___.' >' "".
//         | | :  `- \`.;`\ _ /`;.`/ - ` : | |
//         \  \ `_.   \_ __\ /__ _/   .-` /  /
//     =====`-.____`.___ \_____/___.-`___.-'=====
//                       `=---='
//      Buddha bless your code to be bug free
'''
import keras.backend as K
from keras.models import load_model
import keras.callbacks
from keras.preprocessing import sequence
import subprocess
from ImaGene import *
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import random
import string
class ImaTrain:
    """
    ImaTrain is a keras CNN trainer, written to .
    There are three major functions:
        1.

    Given installed msms jar and msms bash script, ImaTrain can handle simulation and conversion into
    adjusted numpy arrays.

    contact: xiao.liu16@imperial.ac.uk or dxiaomingl@gmail.com
    """
    def __init__(self, model_path = None, hpc = True, onflight = False, save_mem = True, verbose = False):
        if model_path != None:
            self.model = load_model(model_path)
            self.model_name = model_path.split('/')[-1]
        else:
            self.model = None
            self.model_name = None
        self.imanet = None
        self.trainer_ID = ''.join(random.choice(string.ascii_lowercase) for i in range(6))
        self.imagenes = None
        self.hpc = hpc
        self.HOME = ''
        if hpc:
            self.HOME = '/rds/general/user/xml116/home/'
        self.onflight = onflight
        self.save_mem = save_mem
        self.set_callbacks()
        self.script_path = '' #msms.sh bash script path
        self.msms_jar = '' #msms.jar path
        self.sim_file = '' #simulation data file
        self.dem_model = 3 #3 == Marth-3epoch-CEU
        self.sim_mode = '' #Binary/Continuous
        self.epochs = 0 #Number of epochs
        self.sim_num = 0 #Number of simulated data files
        self.gt = [] #B/W genotype matrices
        self.test_gt= None #
        self.labels = [] #Lables
        self.test_labels = None
        self.pos = []
        self.test_pos = None
        self.pos_len = 300
        self.num_chromosomes = 198 #Number of rows
        self.ima_pad = True #Pad SNP matrices to the same size
        self.ima_trans = False #Or transform the image
        self.pos_pad = True #Pad position arrays to the same size
        self.pred = None
        self.batch_size = 64
        self.validation_split = 0.10
        self.epochs_ran = 0
        self.lr_log = [] #Learning rate log
        self.classification_type = 'categorical' #Categorical or binary
        self.pos_type = 'raw' #Using either raw (relative) positions or positional densities
        self.data_array_dir = 'DataArray/' #Directory where read and converted data can be saved in numpy format
        if verbose:
            print('Model name: {0}, hpc: {1}, onflight: {2}'.format(self.model_name,self.hpc,self.onflight))
            if self.imagenes == None:
                print('Please setup ImaGenes or msms simulations')

    def set_sim(self, script_path = 'simulate_5_classes.sh', jar_dir = '$HOME/msms/lib/msms.jar', outfile = '$HOME/sim_5_classes',
                dem_model = 3, sim_mode = 'Binary'):
        self.script_path = script_path
        self.msms_jar = jar_dir
        self.sim_file = outfile
        self.dem_model = dem_model
        self.sim_mode = sim_mode
        if self.onflight:
            create_cmd = ['mkdir', outfile]
            p=subprocess.Popen(create_cmd)
    def simulate(self,num=None):
        if self.onflight:
            simulate_cmd = ['bash', self.script_path, self.msms_jar, self.sim_file, str(self.dem_model), self.sim_mode]
            p = subprocess.Popen(['chmod', '+x', self.script_path])
            p = subprocess.Popen(simulate_cmd)
            p.wait()
        else:
            self.set_sim_num(num)
            for i in range(num):
                print('Simulating set number {0}'.format(i+1))
                sub_simfile = '{0}_{1}'.format(self.sim_file,str(i+1))
                print('Making Simfile: {0}'.format(sub_simfile))
                make_cmd = ['mkdir', sub_simfile]
                p = subprocess.Popen(make_cmd)
                p.wait()
                simulate_cmd = ['bash', self.script_path, self.msms_jar, '{0}'.format(sub_simfile),
                                str(self.dem_model),
                                self.sim_mode]
                p = subprocess.Popen(['chmod', '+x', self.script_path])
                print('Simulating')
                p = subprocess.Popen(simulate_cmd)
                p.wait()
    def sort(self, index=None):
        '''
        Sorts
        :param index: Integer or none.
        :return:
        '''
        if index!=None:
            try:
                for j in range(self.gt[index].shape[0]):
                    self.gt[index][j] = self.sort_min_diff(self.gt[index][j])
            except ValueError:
                self.without_channel(index)
                for j in range(self.gt[index].shape[0]):
                    self.gt[index][j] = self.sort_min_diff(self.gt[index][j])
        else:
            for i in range(len(self.gt)):
                try:
                    for j in range(self.gt[i].shape[0]):
                        self.gt[i][j] = self.sort_min_diff(self.gt[i][j])
                except ValueError:
                    self.without_channel(i)
                    for j in range(self.gt[i].shape[0]):
                        self.gt[i][j] = self.sort_min_diff(self.gt[i][j])
    def sort_min_diff(self, amat):
        '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
        this problem is NP, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
        assumes your input matrix is a numpy array (Schrider)'''
        mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
        v = mb.kneighbors(amat)
        smallest = np.argmin(v[0].sum(axis=1))
        return amat[v[1][smallest]]
    def set_num_chromosomes(self,n):
        self.num_chromosomes = n
    def set_pos_len(self, l):
        self.pos_len = l
    def set_sizes(self,n,l):
        self.set_num_chromosomes(n)
        self.set_pos_len(l)
    def set_sim_file(self, f):
        self.sim_file = f
    def set_classification_type(self,t):
        self.classification_type = t
    def convert_targets(self):
        if self.classification_type == 'categorical':
            self.imagenes.targets = to_categorical(self.imagenes.targets)
        else:
            self.imagenes.targets = to_binary(self.imagenes.targets)

    def set_home(self, HOME):
        self.HOME = HOME
    def set_mode(self,data_mode):
        if data_mode == 1:
            self.onflight = False
            self.save_mem = True
        elif data_mode == 2:
            self.onflight = True
            self.save_mem = False
        elif data_mode == 3:
            self.onflight = False
            self.save_mem = False

    def set_save_mem(self, b):
        '''
        :param b: a boolean
        :return:
        '''
        self.save_mem = b
    def set_callbacks(self, ea_patience=5, lr_patience=3, lr_factor=0.1,
                      ea_monitor='val_acc', lr_monitor='val_loss', model_checkpoint_monitor='val_loss',
                      callbacks=None):
        if callbacks!=None:
            self.callbacks_list = callbacks
        else:
            self.callbacks_list = [
                keras.callbacks.TensorBoard(
                    log_dir='{0}Log_Dir/{1}_{2}'.format(self.HOME, self.model_name, self.trainer_ID),
                    histogram_freq=1,
                ),
                keras.callbacks.EarlyStopping(
                    monitor=ea_monitor,
                    patience=ea_patience,
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath='{0}Models/{1}_{2}_weights.h5'.format(self.HOME, self.model_name, self.trainer_ID),
                    monitor=model_checkpoint_monitor,
                    save_best_only=True,
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor=lr_monitor,
                    factor=lr_factor,
                    patience=lr_patience,
                )
            ]
    def set_data_array_dir(self,darray_dir,make = True):
        if make:
            make_cmd = ['mkdir', darray_dir]
            p = subprocess.Popen(make_cmd)
        self.data_array_dir = darray_dir
    def set_epochs(self, e):
        self.epochs = e
    def set_sim_num(self, num):
        self.sim_num = num
    def set_batch_size(self,b):
        self.batch_size = b
    def set_validation_split(self,v):
        self.validation_split = v
    def read_numpy_dir(self, dir=None, array_num=None):
        '''
        :param dir: None or string of path to directory cotaining numpy arrays
        :param array_num: None or integer of the number of numpy arrays in dir
        :return: None. Appends all numpy arrays into self.gt, self.pos and self.labels
        '''
        file = dir if dir != None else self.data_array_dir
        num = array_num if array_num != None else self.sim_num
        for n in range(num):
            gt_path = '{0}GT_{1}.npy'.format(file,num+1)
            pos_path = '{0}POS_{1}.npy'.format(file,num+1)
            labels_path = '{0}LABEL_{1}.npy'.format(file,num+1)
            self.read_numpy(gt_path,pos_path,labels_path)
    def read_numpy(self, gt = '', pos = '', labels = '', sort=True):
        '''
        :param gt: String: path of numpy file containing genotype matrix (chromosomes on rows and snps on columns)
        :param pos: Stirng: path of numpy file containing position of snps
        :param labels: String: path of file containing targets
        :return: None
        '''
        self.clear_mem()
        try:
            self.gt.append(np.load(gt))
            if sort: self.sort()
        except FileNotFoundError:
            pass
        try:
            self.pos.append(np.load(pos))
        except FileNotFoundError:
            pass
        try:
            self.labels.append(np.load(labels))
        except FileNotFoundError:
            pass
    def read_data(self, sort = True):
        try:
            myfile = ImaFile(simulations_folder=self.sim_file, nr_samples=self.num_chromosomes,
                                 model_name='Marth-3epoch-CEU')
            self.append_data(myfile,sort)
        except EOFError:
            print('Simulation incorrectly compressed. File aborted. Please remake the file.')
            pass
    def read_test(self,gt='',pos='',labels='',simfolder='', type='numpy', sort = True):
        if type=='numpy':
            try:
                self.test_gt = np.load(gt)
                print(self.test_gt.shape)
            except FileNotFoundError:
                pass
            try:
                self.test_pos = np.load(pos)
            except FileNotFoundError:
                pass
            try:
                self.test_labels = np.load(labels)
            except FileNotFoundError:
                pass
        elif type=='msms':
            try:
                myfile = ImaFile(simulations_folder=simfolder, nr_samples=self.num_chromosomes,
                                 model_name='Marth-3epoch-CEU')
                self.append_data(myfile, sort)
            except EOFError:
                print('Simulation incorrectly compressed. File aborted. Please remake the file.')
                pass

    def set_data(self, sort = True):
        '''Unlike read_data, set_data should only be used once
        in off-flight mode where all SimFiles are provided and
        data are read in one go'''
        for i in range(self.sim_num):
            print('Reading sim file: {0}'.format(str(i+1)))
            try:
                myfile = ImaFile(simulations_folder='{0}_{1}'.format(self.sim_file,str(i+1)), nr_samples=self.num_chromosomes,
                                 model_name='Marth-3epoch-CEU')
                # set_data aims to stores all data in memory at once, so save_mem is always false for this method
                self.append_data(myfile,save_mem=False,sort=sort)
            except EOFError:
                print('Simulation incorrectly compressed. File aborted. Please remake the file.')
                pass
    def append_data(self,imafile, save_mem = None, sort=True, verbose=True):
        '''
        Data handling and processing.
        Images will either be transformed or padded by skimage or keras.preprocessing, respectively.
        :param imafile:
        :param save_mem: save_mem==1 will dsicard existing data prior to
        appending new data. Only set it to 1 if you know what you are doing.
        :param sort:
        :param verbose:
        :return:
        '''


        '''
        Some methods in ImaTrain requires append_data to have save_mem==1 or 0;
        while for the most part save_mem==self.save_mem
        '''
        save_mem = save_mem if save_mem != None else self.save_mem
        self.clear_mem(save_mem)
        self.imagenes = imafile.read_simulations(parameter_name='selection_coeff_hetero', max_nrepl=2000)
        '''Shuffling ImaGenes'''

        '''Converting and Appending targets'''
        self.convert_targets()
        '''Padding and extracting genotypes and positions'''
        if self.ima_trans:
            if verbose: print('Transforming genotypes')
            self.imagenes.resize((self.num_chromosomes, self.pos_len))
            self.imagenes.convert(verbose=True)
            rnd_idx = get_index_random(self.imagenes)
            self.imagenes.subset(rnd_idx)
            labels = self.imagenes.targets
            self.labels.append(labels)
            self.gt.append(self.imagenes.data)
        elif self.ima_pad:
            if verbose: print('Padding genotypes')

            print(len(self.imagenes.data))
            images = np.zeros((len(self.imagenes.data),self.num_chromosomes,self.pos_len))
            for i,image in enumerate(self.imagenes.data):
                image = np.asarray(image)
                dim = image.shape
                image = image.reshape((dim[0],dim[1]))
                padded = sequence.pad_sequences(image, padding='post',  maxlen=self.pos_len)
                images[i]=padded
            labels = self.imagenes.targets
            images, labels = shuffle(images,labels)
            images = self.ima_convert(images,verbose=True)
            self.labels.append(labels)
            self.gt.append(images)
        if sort: self.sort(-1)
        if verbose and self.pos_pad: print('padding positions')
        if self.pos_type == 'raw':
            if self.pos_pad: self.pos.append(sequence.pad_sequences(self.imagenes.positions, padding='post',
                                                                value=-1, dtype='float32', maxlen=self.pos_len))
            elif self.pos_type == 'density':
                self.pos.append(self.pos_to_density(self.imagenes.positions, self.pos_len))

    def ima_convert(self, data, normalise=False, flip=True, verbose=False):
        """
        Check for correct data type and convert otherwise. Convert to float numpy arrays [0,1] too. If flip true, then flips 0-1
        """
        # if list, put is as numpy array
        if type(data) == list:
            if len(np.unique(self.dimensions[0])) * len(np.unique(self.dimensions[1])) == 1:
                if verbose:
                    print('Converting to numpy array.')
                self.data = np.asarray(self.data)
            else:
                print('Aborted. All images must have the same shape.')
                return 1
        # if unit8, put it as float and divide by 255
        if data.dtype == 'uint8':
            if verbose:
                print('Converting to float32.')
            data = data.astype('float32')
        if data.max() > 1:
            if verbose:
                print('Converting to [0,1].')
            data /= 255.
        # normalise
        if normalise == True:
            if verbose:
                print('Normalising samplewise.')
            for i in range(len(data)):
                mean = data[i].mean()
                std = data[i].std()
                data[i] -= mean
                data[i] /= std
        # flip
        if flip == True:
            if verbose:
                print('Flipping values.')
            for i in range(len(data)):
                data[i] = 1. - data[i]
        if verbose:
            print('A numpy array with dimensions', data.shape)
        return data
    def with_channel(self,i):
        dim = self.gt[i].shape
        new_dim = []
        for item in dim:
            new_dim.append(item)
        new_dim.append(1)
        self.gt[i] = self.gt[i].reshape(new_dim)
    def without_channel(self,i):
        new_dim = list(self.gt[i].shape)
        new_dim.pop(-1)
        self.gt[i] = self.gt[i].reshape(new_dim)
    def adjust_channel(self,i):
        if self.gt[i].shape[-1] == 1:
            self.without_channel(i)
        else:
            self.with_channel(i)
        print('Image adjuted')
    def convert_to_numpy(self,sort=True):
        '''
        Convert msms simulations in self.sim_file(s) to numpy arrays
        and save them to DataArray path. This method assumes the memory
        to be limited and therefore only reads and stores one set
        of simulation at any time.
        :return: None
        '''
        num = self.sim_num if self.sim_num > 0 else 1
        for i in range(num):
            print('Reading and converting sim file: {0}'.format(str(i+1)))
            try:
                myfile = ImaFile(simulations_folder='{0}_{1}'.format(self.sim_file,str(i+1)), nr_samples=self.num_chromosomes,
                                 model_name='Marth-3epoch-CEU')
                self.append_data(myfile,save_mem=True,sort=sort) #Always only keep one set of data in memeory during conversion
                np.save('{0}GT_{1}.npy'.format(self.data_array_dir,str(i+1)),self.gt[0])
                np.save('{0}POS_{1}.npy'.format(self.data_array_dir, str(i + 1)), self.pos[0])
                np.save('{0}LABEL_{1}.npy'.format(self.data_array_dir, str(i + 1)), self.labels[0])
            except EOFError:
                print('Simulation incorrectly compressed. File number {0} aborted. Please remake the file'.format(i+1))

    def save_data(self):
        '''
        This method saves all arrray data in memory into numpy files
        :return: None
        '''
        for i in range(self.sim_num):
            print('Saving read data set: {0}'.format(str(i+1)))
            np.save('{0}GT_{1}.npy'.format(self.data_array_dir,str(i+1)),self.gt[i])
            np.save('{0}POS_{1}.npy'.format(self.data_array_dir, str(i + 1)),self.pos[i])
            np.save('{0}LABEL_{1}.npy'.format(self.data_array_dir, str(i + 1)),self.labels[i])
    def pos_to_density(self, positions, parts=128):
        '''
        :param positions: array of relative positions
        :param parts: the length of the density array
        :return: an array of positional density
        '''
        length = len(positions)
        pos_density_array = np.zeros((length, parts))
        for i in range(length):
            pos_density = np.zeros(parts)
            for position in positions[i:]:
                pos_density[(position * parts).astype(int) - 1] += 1
            pos_density_array[i] = pos_density
        return pos_density_array

    def train(self,e,num,mode = '1b_image', sort = True, verbose = True):
        self.set_epochs(e)
        self.set_sim_num(num)
        mynet = ImaNet(name=self.model_name + '_net')
        self.imanet = mynet
        if verbose: print('Model Summary {}, ID: {}'.format(self.model.summary(),self.trainer_ID))
        for i in range(self.sim_num):
            sample_counter = i
            if verbose: print('Training on sample {0}'.format(sample_counter + 1))
            if self.onflight:
                if verbose: print('Simulating data set {0}'.format(i+1))
                self.simulate()
                self.read_data(sort)
                '''There is only one ImaGene during on-flight training, 
                because previous ImaGenes are replaced after each new simulation'''
                #i = 0
            elif self.save_mem:
                if mode == 'multi':
                    self.read_numpy(gt='{0}GT_{1}.npy'.format(self.data_array_dir,i+1),
                                    pos='{0}POS_{1}.npy'.format(self.data_array_dir,i+1),
                                    labels='{0}LABEL_{1}.npy'.format(self.data_array_dir,i+1),sort=sort)
                elif mode == '1b_image':
                    self.read_numpy(gt='{0}GT_{1}.npy'.format(self.data_array_dir,i+1),
                                    labels='{0}LABEL_{1}.npy'.format(self.data_array_dir,i+1),sort=sort)
                elif mode == '1b_pos':
                    self.read_numpy(pos='{0}POS_{1}.npy'.format(self.data_array_dir,i+1),
                                    labels='{0}LABEL_{1}.npy'.format(self.data_array_dir,i+1))
                '''
                It's important that i=0 in save memory mode because there is only one set of data
                stored in the arrays at any time, as the previous array is popped out after training's
                finished on that data set.
                '''
                i = 0
            for e in range(self.epochs):
                if verbose: print('Training on epoch {0}, sample {1}.'.format(e + 1,sample_counter+1))
                if mode == 'multi':
                    try:
                        score = self.model.fit([self.gt[i], self.pos[i]], self.labels[i],
                                               batch_size=self.batch_size,
                                               epochs=1,
                                               verbose=1,
                                               validation_split=self.validation_split,
                                               callbacks=self.callbacks_list)
                    except ValueError:
                        '''
                        Value errors frequently occur, especially in the case of training
                        Conv1D and Conv2D nets
                        '''
                        self.adjust_channel(i)
                        score = self.model.fit([self.gt[i], self.pos[i]], self.labels[i],
                                               batch_size=self.batch_size,
                                               epochs=1,
                                               verbose=1,
                                               validation_split=self.validation_split,
                                               callbacks=self.callbacks_list)
                    self.imanet.update_scores(score)
                    self.check_point(self.imanet)
                elif mode == '1b_image':
                    try:
                        score = self.model.fit(self.gt[i], self.labels[i],
                                               batch_size=self.batch_size,
                                               epochs=1,
                                               verbose=1,
                                               validation_split=self.validation_split,
                                               callbacks=self.callbacks_list)
                    except ValueError:
                        self.adjust_channel(i)
                        score = self.model.fit(self.gt[i], self.labels[i],
                                               batch_size=self.batch_size,
                                               epochs=1,
                                               verbose=1,
                                               validation_split=self.validation_split,
                                               callbacks=self.callbacks_list)
                    self.imanet.update_scores(score)
                    self.check_point(self.imanet)
                elif mode == '1b_pos':
                    score = self.model.fit(self.pos[i], self.labels[i],
                                           batch_size=self.batch_size,
                                           epochs=1,
                                           verbose=1,
                                           validation_split=self.validation_split,
                                           callbacks=self.callbacks_list)
                    self.imanet.update_scores(score)
                    self.check_point(self.imanet)
                else:
                    return 'Please input the correct input format - multi, 1b_image or 1b_pos'
        self.check_point(self.imanet, patience=1)
        return self.imanet
    def check_point(self, mynet, patience = 10,verbose = True):
        self.epochs_ran += 1
        if self.epochs_ran % patience == 0:
            mynet.save('{0}{1}_{2}_net'.format(self.HOME, self.model_name, self.trainer_ID))
            mynet.plot_train('{0}{1}_{2}_Training_History.png'.format(self.HOME, self.model_name,self.trainer_ID))
            plt.close()
            '''
            Appending current learning rate
            '''
            lr = K.eval(self.model.optimizer.lr)
            self.lr_log.append(lr)
            if verbose: print('Mynet and plot saved. Current learning rate: {0}. Trainer ID: {1}'.format(lr,self.trainer_ID))
    def evaluate(self,test_gt = None, test_pos = None,test_labels=None):
        test_gt = test_gt if test_gt != None else self.test_gt
        test_pos = test_pos if test_pos != None else self.test_pos
        test_labels = test_labels if test_labels != None else self.test_labels
        print(type(test_gt))
        print(type(test_pos))
        print(type(test_labels))
        print(type(self.test_gt))
        print(type(self.test_pos))
        print(type(self.test_labels))
        if isinstance(test_pos,np.ndarray):
            try:
                return self.model.evaluate([test_gt, test_pos],test_labels,batch_size=None)
            except ValueError:
                dim = list(test_gt.shape)
                print(dim)
                dim.pop() if dim[-1]==1 else dim.append(1)
                test_gt = test_gt.reshape(dim)
                return self.model.evaluate([test_gt, test_pos], test_labels, batch_size=None)
        else:
            try:
                return self.model.evaluate(test_gt,test_labels,batch_size=None)
            except ValueError:
                dim = list(test_gt.shape)
                dim.pop() if dim[-1]==1 else dim.append(1)
                test_gt = test_gt.reshape(dim)
                return self.model.evaluate(test_gt, test_labels, batch_size=None)
    def predict(self,gt=None,pos=None):
        gt = gt if gt != None else self.test_gt
        pos = pos if pos != None else self.test_pos
        if isinstance(pos,np.ndarray):
            try:
                self.pred = self.model.predict([gt,pos], batch_size=None)
            except ValueError:
                dim = list(gt.shape)
                dim.pop() if dim[-1] == 1 else dim.append(1)
                gt = gt.reshape(dim)
                self.pred = self.model.predict([gt,pos], batch_size=None)
        else:
            try:
                self.pred = self.model.predict(gt,batch_size=None)
            except ValueError:
                dim = list(gt.shape)
                dim.pop() if dim[-1] == 1 else dim.append(1)
                gt = gt.reshape(dim)
                self.pred = self.model.predict(gt, batch_size=None)
        #print(self.pred)
        return self.pred
    def plot_cm(self, file=None):

        probs = self.pred
        classes = np.arange(probs.shape[1])
        #targets = np.arange(probs.shape[1])
        values = np.zeros((3, self.test_gt.shape[0]), dtype='float32')

        values[1, :] = classes[np.argmax(probs, axis=1)]
        values[0, :] = classes[np.argmax(self.test_labels, axis=1)]
        values[2, :] = [np.average(classes, weights=probs[i]) for i in range(probs.shape[0])]



        cm = confusion_matrix(values[0, :], values[1, :])
        accuracy = np.trace(cm) / float(np.sum(cm))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=8)
        plt.yticks(tick_marks, classes, fontsize=8)

        thresh = cm.max() / 1.5
        for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}'.format(accuracy))
        #plt.show()

        if (file == None):
            plt.show()
        else:
            plt.savefig(file)
        return 0


    def clear_mem(self,save_mem=None):
        save_mem = save_mem if save_mem != None else self.save_mem
        if save_mem:
            if len(self.gt) > 0:
                self.gt.pop()
            if len(self.pos) > 0:
                self.pos.pop()
            if len(self.labels) > 0:
                self.labels.pop()
    def get_lr(self, all = False):
        '''
        :return: if all==1, a list of learning rate history;
        else if all==0,  a float of current learning rate
        '''
        if all:
            return self.lr_log
        return self.lr_log[-1]
    def fit_input(self,model):
        pass
    def save_image(self, file, index1, index2):
        data = self.gt[index1][index2]
        plt.imshow(data,cmap='Greys',interpolation='nearest')
        plt.savefig('{0}_{1}{2}.png'.format(file,index1,index2))
    def save(self, file):
        """
        Save to file
        """
        with open(file,'wb') as fp:
            pickle.dump(self, fp)
        return 0
    def __str__(self):
        return 'On-flight: {0}, model name: {1}, model summary: {2}'\
            .format(self.onflight,self.model_name, self.model.summary())

class ImaTrainHPC(ImaTrain):
    def __init__(self, user, model_path = None, hpc = True, onflight = True, verbose = False):
        super.__init__(model_path, hpc, onflight, verbose)
        self.ROOT = 'rds/general/user/'
        self.USER = user
        self.HOME = '{0}{1}/home/'.format(self.ROOT,self.USER)
        self.EPHEMERAL = '{0}{1}/ephemeral/'.format(self.ROOT,self.USER)
    def change_root(self,root,update=True):
        self.ROOT = root
        if update:
            self.HOME = '{0}{1}/home/'.format(self.ROOT, self.USER)
            self.EPHEMERAL = '{0}{1}/ephemeral/'.format(self.ROOT, self.USER)
    def change_user(self,user,update=True):
        self.USER = user
        if update:
            self.HOME = '{0}{1}/home/'.format(self.ROOT, self.USER)
            self.EPHEMERAL = '{0}{1}/ephemeral/'.format(self.ROOT, self.USER)

'''
class HyperImaTrain(ImaTrain):
    def __init__(self):
        super.__init__()
        return None

'''