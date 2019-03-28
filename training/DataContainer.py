import numpy  as np

class DataContainer:
    def __repr__(self):
         return "DataContainer"
    def __init__(self, filename):
        #read in data
        dictionary = np.load(filename)
        #number of atoms
        if 'N' in dictionary: 
            self._N = dictionary['N'] 
        else:
            self._N = None
        #atomic numbers/nuclear charges
        if 'Z' in dictionary: 
            self._Z = dictionary['Z'] 
        else:
            self._Z = None
        #reference dipole moment vector
        if 'D' in dictionary: 
            self._D = dictionary['D'] 
        else:
            self._D = None
        #reference total charge
        if 'Q' in dictionary: 
            self._Q = dictionary['Q'] 
        else:
            self._Q = None
        #reference atomic charges
        if 'Qa' in dictionary: 
            self._Qa = dictionary['Qa'] 
        else:
            self._Qa = None
        #positions (cartesian coordinates)
        if 'R' in dictionary:     
            self._R = dictionary['R'] 
        else:
            self._R = None
        #reference energy
        if 'E' in dictionary:
            self._E = dictionary['E'] 
        else:
            self._E = None
        #reference atomic energies
        if 'Ea' in dictionary:
            self._Ea = dictionary['Ea']
        else:
            self._Ea = None
        #reference forces
        if 'F' in dictionary:
            self._F = dictionary['F'] 
        else:
            self._F = None

        #maximum number of atoms per molecule
        self._N_max    = self.Z.shape[1] 

        #construct indices used to extract position vectors to calculate relative positions 
        #(basically, constructs indices for calculating all possible interactions (excluding self interactions), 
        #this is a naive (but simple) O(N^2) approach, could be replaced by something more sophisticated) 
        self._idx_i = np.empty([self.N_max, self.N_max-1],dtype=int)
        for i in range(self.idx_i.shape[0]):
            for j in range(self.idx_i.shape[1]):
                self._idx_i[i,j] = i

        self._idx_j = np.empty([self.N_max, self.N_max-1],dtype=int)
        for i in range(self.idx_j.shape[0]):
            c = 0
            for j in range(self.idx_j.shape[0]):
                if j != i:
                    self._idx_j[i,c] = j
                    c += 1

    @property
    def N_max(self):
        return self._N_max

    @property
    def N(self):
        return self._N

    @property
    def Z(self):
        return self._Z

    @property
    def Q(self):
        return self._Q

    @property
    def Qa(self):
        return self._Qa

    @property
    def D(self):
        return self._D

    @property
    def R(self):
        return self._R

    @property
    def E(self):
        return self._E

    @property
    def Ea(self):
        return self._Ea
    
    @property
    def F(self):
        return self._F

    #indices for atoms i (when calculating interactions)
    @property
    def idx_i(self):
        return self._idx_i

    #indices for atoms j (when calculating interactions)
    @property
    def idx_j(self):
        return self._idx_j

    def __len__(self): 
        return self.Z.shape[0]

    def __getitem__(self, idx):
        if type(idx) is int or type(idx) is np.int64:
            idx = [idx]

        data = {'E':         [],
                'Ea':        [],    
                'F':         [],
                'Z':         [],
                'D':         [],
                'Q':         [],
                'Qa':        [],
                'R':         [],
                'idx_i':     [],
                'idx_j':     [],
                'batch_seg': [],
                'offsets'  : []
                }

        Ntot = 0 #total number of atoms
        Itot = 0 #total number of interactions
        for k, i in enumerate(idx):
            N = self.N[i] #number of atoms
            I = N*(N-1)   #number of interactions
            #append data
            if self.E is not None:
                data['E'].append(self.E[i])
            else:
                data['E'].append(np.nan)
            if self.Ea is not None:
                data['Ea'].extend(self.Ea[i,:N].tolist())
            else:
                data['Ea'].extend([np.nan])
            if self.Q is not None:
                data['Q'].append(self.Q[i])
            else:
                data['Q'].append(np.nan)
            if self.Qa is not None:
                data['Qa'].extend(self.Qa[i,:N].tolist())
            else:
                data['Qa'].extend([np.nan])
            if self.Z is not None:
                data['Z'].extend(self.Z[i,:N].tolist())
            else:
                data['Z'].append(0)
            if self.D is not None:
                data['D'].extend(self.D[i:i+1,:].tolist())
            else:
                data['D'].extend([[np.nan,np.nan,np.nan]])
            if self.R is not None:
                data['R'].extend(self.R[i,:N,:].tolist())
            else:
                data['R'].extend([[np.nan,np.nan,np.nan]])
            if self.F is not None:
                data['F'].extend(self.F[i,:N,:].tolist())
            else:
                data['F'].extend([[np.nan,np.nan,np.nan]])
            data['idx_i'].extend(np.reshape(self.idx_i[:N,:N-1]+Ntot,[-1]).tolist())
            data['idx_j'].extend(np.reshape(self.idx_j[:N,:N-1]+Ntot,[-1]).tolist())
            #offsets could be added in case they are need
            data['batch_seg'].extend([k] * N)
            #increment totals
            Ntot += N
            Itot += I

        return data

