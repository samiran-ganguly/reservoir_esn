# ESN Computing Library built on top of TensorFlow
# Copyright (C) 2018  Samiran Ganguly, University of Virginia, sganguly@virginia.edu
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
# 
#

import TensorFlow as tf

class ESN:
    def __init__(self, sName, nResSize = 1, nInSize = 1, nOutSize = 1, sType = 'tanh'):
        self.sName = sName
        # Set the ESN sizes
        self.nResSize = nResSize
        self.nInSize = nInSize
        self.nOutSize = nOutSize
        # Create the network nodes:
        # - self stores the internal states 
        #- inp stores the incoming dataframes
        # - outp stores the outgoing dataframes
        self.fState = tf.get_variable('state',[nResSize],dtype=tf.float64,initializer=tf.zeros_initializer)
        self.fInp = tf.get_variable('inp',[1+nInSize],dtype=tf.float64,initializer=tf.zeros_initializer)
        self.fOutp = tf.get_variable('outp',[nOutSize],dtype=tf.float64,initializer=tf.zeros_initializer)
        self.fStateMatrix = tf.get_variable('state_matrix',[nInSizenResSize+1],dtype=tf.float64,initializer=tf.zeros_initializer)
        # Set the interconnection matrices between the input and output to the ESN
        self.Win = tf.get_variable('Win', [nResSize,nInSize+1],dtype=tf.float64,initializer=tf.RandomUniform(0,1))
        self.Wfb = tf.get_variable('Wfb', [nResSize,nOutSize],dtype=tf.float64,initializer=tf.RandomUniform(0,1))
        self.Wout = tf.get_variable('Wout', [nOutSize,nResSize],dtype=tf.float64,initializer=tf.RandomUniform(0,1))
        #  Set the ESN connections
        self.Wself = tf.get_variable('Wself', [nResSize,nResSize],dtype=tf.float64,initializer=tf.RandomUniform(0,1))
        # Set the ESN node activation
        self.sType = sType
        # Set Scaling
        self.fActScale = fActScale
        self.fDecayRate = fDecayRate
        self.fNoiseScale = fNoiseScale
        
    def update_state(self):
        try:
            fAct = tf.matmul(self.Win,self.fInp) + tf.matmul(self.Wfb,self.fState) + tf.matmul(self.Wfb,self.fOutp)
            fDecay = (1-self.fDecayRate)*self.fState
            fNoise = tf.random_uniform([self.nResSize],-1,1,dtype=tf.float64)
        except ArithmeticError:
            tf.logging.log(tf.logging.ERROR,'ESN '+self.sName+': Computational error trying to calculate node argument')
            print('ESN '+self.sName+': Computational error trying to calculate node argument')
        try:
            if self.sType == 'tanh':
                self.fState = tf.add(fDecay,tf.add(tf.scalar_mul(self.fActScale,tf.tanh(fAct)),tf.scalar_mul(self.fNoiseScale,fNoise)))
            elif (self.sType == 'sigmoid') or (self.sType == 'exp'):
                self.fState = tf.add(fDecay,tf.add(tf.scalar_mul(self.fActScale,tf.sigmoid(fAct)),tf.scalar_mul(self.fNoiseScale,fNoise)))
            else:
                self.fState = tf.add(fDecay,tf.add(tf.scalar_mul(self.fActScale,fAct),tf.scalar_mul(self.fNoiseScale,fNoise)))
        except ArithmeticError:
            tf.logging.log(tf.logging.ERROR,'ESN '+self.sName+': Computational error trying to calculate node activation')
            print('ESN '+self.sName+': Computational error trying to calculate node activation')
        try:
            fStateVec = tf.concat(self.fInp,self.fState,1)
            self.fOutp = tf.matmul(self.Wout,self.fState)
        except ArithmeticError:
            tf.logging.log(tf.logging.ERROR,'ESN '+self.sName+': Computational error trying to calculate network output')
            print('ESN '+self.sName+': Computational error trying to calculate network output')
        
        
    def apply_input(self,fUt = 0,fInputScaler = 0,fInputShifter = 0):
        try:
            self.fInp = tf.concat(tf.convert_to_tensor(1.0,dtype=tf.float64), add(tf.scalar_mul(fInputScaler,float64(fUt))+fInputShifter),1);
        except AttributeError:
            tf.logging.log(tf.logging.ERROR,'ESN '+self.sName+': Input dataframe size does not match ESN input node size')
            print('ESN '+self.sName+': Input dataframe size does not match ESN input node size')
    
    def obtain_output(self):
        return self.fOutp
        
    def collect_state_matrix(self):
        try:
            fStateVec = tf.concat(self.fInp,self.fState,1)
            self.fStateMatrix = tf.concat(fStateVec,self.fStateMatrix,0)
        except ArithmeticError:
            tf.logging.log(tf.logging.ERROR,'ESN '+self.sName+': Computational error trying to collect state matrix')
            print('ESN '+self.sName+': Computational error trying to collect state matrix')
        except MemoryError:
            tf.logging.log(tf.logging.ERROR,'ESN '+self.sName+': Ran out of memory trying to collect all the states')
            print('ESN '+self.sName+': Ran out of memory trying to collect all the states')
        
    def get_state_matrix(self):
        return self.fStateMatrix
    
    def pinv(a,rcond=1e-15):
        # Compute the SVD of the input matrix A
        s, u, v = tf.svd(a)
        # Ignore singular values close to zero to prevent numerical overflow
        limit = rcond * tf.reduce_max(s)
        non_zero = tf.greater(s, limit)
        reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(s.shape))
        lhs = tf.matmul(v, tf.matrix_diag(reciprocal))
        return tf.matmul(lhs, u, transpose_b=True)
        
    def adjust_output_weight(self,fTarget):
        try:
            Wout = tf.matmul(fTarget,pinv(get_state_matrix()))
        except ArithmeticError:
            tf.logging.log(tf.logging.ERROR,'ESN '+self.sName+': Computational error trying to calculate Wout')
            print('ESN '+self.sName+': Computational error trying to calculate Wout')
        return Wout
        
    def train_ESN(self):
        
        