import tensorflow  as tf
import threading

class DataQueue:
    def __repr__(self):
         return "DataQueue"
    def __init__(self, get_data, capacity=5000, scope=None, dtype=tf.float32):
        self._get_data = get_data
        self._is_running = False
        self._scope = scope

        with tf.variable_scope(self.scope):
            dtypes = [ dtype,    dtype,    dtype,    tf.int32, dtype,    dtype,    dtype,    dtype,    tf.int32, tf.int32, tf.int32 ]
            shapes = [ [None, ], [None, ], [None,3], [None, ], [None,3], [None, ], [None, ], [None,3], [None, ], [None, ], [None, ] ]   
            
            #define placeholders
            self._E          = tf.placeholder(dtypes[ 0], shape=shapes[ 0], name="E")
            self._Ea         = tf.placeholder(dtypes[ 1], shape=shapes[ 1], name="Ea")
            self._F          = tf.placeholder(dtypes[ 2], shape=shapes[ 2], name="F")
            self._Z          = tf.placeholder(dtypes[ 3], shape=shapes[ 3], name="Z")
            self._D          = tf.placeholder(dtypes[ 4], shape=shapes[ 4], name="D")      
            self._Q          = tf.placeholder(dtypes[ 5], shape=shapes[ 5], name="Q")  
            self._Qa         = tf.placeholder(dtypes[ 6], shape=shapes[ 6], name="Qa")  
            self._R          = tf.placeholder(dtypes[ 7], shape=shapes[ 7], name="R")      
            self._idx_i      = tf.placeholder(dtypes[ 8], shape=shapes[ 8], name="idx_i") 
            self._idx_j      = tf.placeholder(dtypes[ 9], shape=shapes[ 9], name="idx_j") 
            self._batch_seg  = tf.placeholder(dtypes[10], shape=shapes[10], name="batch_seg") 
            placeholders =  [ self.E, self.Ea, self.F, self.Z, self.D, self.Q, self.Qa, self.R, self.idx_i, self.idx_j, self.batch_seg]

            self._queue  = tf.PaddingFIFOQueue(capacity=capacity, dtypes=dtypes, shapes=shapes, name="queue")
            self._enqueue_op = self.queue.enqueue(placeholders)
            self._dequeue_op = self.queue.dequeue()

    def create_thread(self, sess, coord=None, daemon=False):
        if coord is None:
            coord = tf.train.Coordinator()

        if self.is_running:
            return []

        thread = threading.Thread(target=self._run, args=(sess, coord))
        thread.daemon = daemon
        thread.start()
        self._is_running = True
        return [thread]

    def _run(self, sess, coord):
        while not coord.should_stop():
            data = self.get_data()
            feed_dict = {
                self.E:  data["E"],
                self.Ea: data["Ea"], 
                self.F:  data["F"], 
                self.Z:  data["Z"], 
                self.D:  data["D"], 
                self.Q:  data["Q"],
                self.Qa: data["Qa"], 
                self.R:  data["R"],
                self.idx_i: data["idx_i"],
                self.idx_j: data["idx_j"],
                self.batch_seg: data["batch_seg"]
            }
            try:
                sess.run(self.enqueue_op, feed_dict=feed_dict)
            except Exception as e:
                coord.request_stop(e)

    @property
    def E(self):
        return self._E

    @property
    def Ea(self):
        return self._Ea

    @property
    def F(self):
        return self._F
    
    @property
    def Z(self):
        return self._Z

    @property
    def D(self):
        return self._D

    @property
    def Q(self):
        return self._Q

    @property
    def Qa(self):
        return self._Qa

    @property
    def R(self):
        return self._R

    @property
    def idx_i(self):
        return self._idx_i
    
    @property
    def idx_j(self):
        return self._idx_j
    
    @property
    def batch_seg(self):
        return self._batch_seg
    
    @property
    def offsets(self):
        return self._offsets

    @property
    def scope(self):
        return self._scope

    @property
    def queue(self):
        return self._queue
    
    @property
    def enqueue_op(self):
        return self._enqueue_op
    
    @property
    def dequeue_op(self):
        return self._dequeue_op
    
    @property
    def get_data(self):
        return self._get_data
    
    @property
    def is_running(self):
        return self._is_running

    
    