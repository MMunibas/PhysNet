import numpy  as np

class DataProvider:
    def __repr__(self):
         return "DataProvider"
    def __init__(self, data, ntrain, nvalid, batch_size=1, valid_batch_size=1, seed=None):
        self._data = data
        self._ndata  = len(data)
        self._ntrain = ntrain
        self._nvalid = nvalid
        self._ntest  = len(data)-self.ntrain-self.nvalid
        self._batch_size = batch_size
        self._valid_batch_size = valid_batch_size

        #random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=seed)

        #create shuffled list of indices
        idx = self._random_state.permutation(np.arange(len(self.data)))

        #store indices of training, validation and test data
        self._idx_train = idx[0:self.ntrain]
        self._idx_valid = idx[self.ntrain:self.ntrain+self.nvalid]
        self._idx_test  = idx[self.ntrain+self.nvalid:]

        #initialize mean/stdev of properties to None, only get calculated if requested
        self._EperA_mean  = None
        self._EperA_stdev = None
        self._FperA_mean  = None
        self._FperA_stdev = None
        self._DperA_mean  = None
        self._DperA_stdev = None

        #for retrieving batches
        self._idx_in_epoch = 0 
        self._valid_idx = 0


    @property
    def data(self):
        return self._data

    @property
    def ndata(self):
        return self._ndata

    @property
    def ntrain(self):
        return self._ntrain
    
    @property
    def nvalid(self):
        return self._nvalid
    
    @property
    def ntest(self):
        return self._ntest

    @property
    def random_state(self):
        return self._random_state

    @property
    def idx_train(self):
        return self._idx_train

    @property
    def idx_valid(self):
        return self._idx_valid   

    @property
    def idx_test(self):
        return self._idx_test

    @property
    def idx_in_epoch(self):
        return self._idx_in_epoch

    @property
    def valid_idx(self):
        return self._valid_idx

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def valid_batch_size(self):
        return self._valid_batch_size

    def _compute_E_statistics(self):
        self._EperA_mean  = 0.0
        self._EperA_stdev = 0.0
        for i in range(self.ntrain):
            tmp = self.get_data(self.idx_train[i])
            m_prev = self.EperA_mean
            x = tmp['E'][0]/(np.shape(tmp['Z'])[0])
            self._EperA_mean  += (x - self._EperA_mean)/(i+1)
            self._EperA_stdev += (x - self._EperA_mean) * (x - m_prev)
        self._EperA_stdev = np.sqrt(self._EperA_stdev/self.ntrain)
        return
    
    @property 
    def EperA_mean(self): #mean energy per atom in the training set
        if self._EperA_mean is None:
            self._compute_E_statistics()
        return self._EperA_mean

    @property
    def EperA_stdev(self): #stdev of energy per atom in the training set
        if self._EperA_stdev is None:
            self._compute_E_statistics()
        return self._EperA_stdev

    def _compute_F_statistics(self):
        self._FperA_mean  = 0.0
        self._FperA_stdev = 0.0
        for i in range(self.ntrain):
            tmp = self.get_data(self.idx_train[i])
            F = tmp["F"]
            x = 0.0
            for i in range(len(F)):
                x += np.sqrt(F[i][0]**2 + F[i][1]**2 + F[i][2]**2)
            m_prev = self.FperA_mean
            x /= len(F)
            self._FperA_mean  += (x - self._FperA_mean)/(i+1)
            self._FperA_stdev += (x - self._FperA_mean) * (x - m_prev)
        self._FperA_stdev = np.sqrt(self._FperA_stdev/self.ntrain)
        return

    @property 
    def FperA_mean(self): #mean force magnitude per atom in the training set
        if self._FperA_mean is None:
            self._compute_F_statistics()
        return self._FperA_mean

    @property
    def FperA_stdev(self): #stdev of force magnitude per atom in the training set
        if self._FperA_stdev is None:
            self._compute_F_statistics()
        return self._FperA_stdev

    def _compute_D_statistics(self):
        self._DperA_mean  = 0.0
        self._DperA_stdev = 0.0
        for i in range(self.ntrain):
            tmp = self.get_data(self.idx_train[i])
            D = tmp["D"]
            x = np.sqrt(D[0]**2 + D[1]**2 + D[2]**2)
            m_prev = self.DperA_mean
            self._DperA_mean  += (x - self._DperA_mean)/(i+1)
            self._DperA_stdev += (x - self._DperA_mean) * (x - m_prev)
        self._DperA_stdev = np.sqrt(self._DperA_stdev/self.ntrain)
        return

    @property
    def DperA_mean(self): #mean partial charge per atom in the training set
        if self._DperA_mean is None:
            self._compute_D_statistics()
        return self._DperA_mean

    @property
    def DperA_stdev(self): #stdev of partial charge per atom in the training set
        if self._DperA_stdev is None:
            self._compute_D_statistics()
        return self._DperA_stdev

    #shuffle the training data
    def shuffle(self):
        self._idx_train = self.random_state.permutation(self.idx_train)

    #returns a batch of samples from the training set
    def next_batch(self):
        start = self.idx_in_epoch
        self._idx_in_epoch += self.batch_size
        #epoch is finished, set needs to be shuffled
        if self.idx_in_epoch > self.ntrain:
            self.shuffle()
            start = 0
            self._idx_in_epoch = self.batch_size
        end = self.idx_in_epoch   
        return self.data[self.idx_train[start:end]]

    #returns a batch of samples from the validation set
    def next_valid_batch(self):
        start = self.valid_idx
        self._valid_idx += self.valid_batch_size
        #finished one pass-through, reset index
        if self.valid_idx > self.nvalid:
            start = 0
            self._valid_idx = self.valid_batch_size
        end =  self.valid_idx
        return self.data[self.idx_valid[start:end]]

    def get_data(self, idx):
        return self.data[idx]

    def get_train_data(self, i):
        idx = self.idx_train[i]
        return self.data[idx]

    def get_all_train_data(self):
        return self.data[self.idx_train]

    def get_valid_data(self, i):
        idx = self.idx_valid[i]
        return self.data[idx]

    def get_all_valid_data(self):
        return self.data[self.idx_valid]

    def get_test_data(self, i):
        idx = self.idx_test[i]
        return self.data[idx]
    
    def get_all_test_data(self):
        return self.data[self.idx_test]