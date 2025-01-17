"""
Implement static and plastic synapses

A synapse object should implement three methods:

- A `__call__` method taking an arbitrary number of inputs and returning
  the synapse output
- An update(xo) method that passes the output of the neurons
- A reset() method

Update and reset can be dummies, but the SpikingNet object expects that
they will exist.

"""

import numpy as np
from .transforms import LowPass, PassThrough
from .trace import Trace
import math
import scipy.interpolate
import warnings
from sklearn.feature_extraction.image import extract_patches_2d

import torch.nn.functional as torchfunc
import torch

class BaseSynapse:
    """
    Base synapse, where synaptic weights are stored in a 2D array.
    """

    def __init__(self, Ne, No, W0, transform=None):
        """
        Parameters
        ----------

        Ne : int
            number of presynaptic neurons
        No : int
            number of postsynaptic neurons
        W0 : np.array
            a 2D array with the initial synaptic weights
        transforms : input transform
        """

        self.Ne = Ne
        self.No = No
        self._W = W0.copy()
        self.has_transform = transform is not None
        if self.has_transform:
            self.transform = transform
        else:
            self.transform = lambda x:  x

    def __call__(self, xe):
        return self.W @ self.transform(xe)

    def reset(self):
        if self.has_transform:
            self.transform.reset()

    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, W):
        self._W = W

    def update(self, pos, learn=True):
        pass


class LambdaSynapse:
    """
    Lambda synapse, where synaptic weights is represented by a function (eg. for procedurally generated connections).
    """

    def __init__(self, Ne, No, W0, transform=None):
        """
        Parameters
        ----------

        Ne : int
            number of presynaptic neurons
        No : int
            number of postsynaptic neurons
        W0 : function
            a function object that applies the synaptic connections
        transforms : input transform
        """

        self.Ne = Ne
        self.No = No
        self._W = W0
        self.has_transform = transform is not None
        if self.has_transform:
            self.transform = transform
        else:
            self.transform = lambda x:  x

    def __call__(self, xe):
        return self.W(self.transform(xe))

    def reset(self):
        if self.has_transform:
            self.transform.reset()

    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, W):
        self._W = W

    def update(self, pos, learn=True):
        pass


class OneToOneSynapse(BaseSynapse):

    def __call__(self, xe):
        return self.W * self.transform(xe)



class PlasticSynapse(BaseSynapse):
    """Simple plastic synapse implementing STDP

    """

    def __init__(self, Ne, No, W0, tre, tro, transform=None,
        rule_params = None, Wlim=1, syn_type=None, tracelim=10):
        """
        Parameters
        ----------

        Ne : int
            number of presynaptic neurons
        No : int
            number of postsynaptic neurons
        W0 : np.array
            a 2D array with the initial synaptic weights
        tre : tuple
            presynaptic trace tuple
        tro : tuple 
            postsynaptic trace tuple
        transform : None | transform
            transform function applied to inputs
        rule_params:
            parameters defining synaptic plasticity rule
        Wlim : float
            clamping parameter for synaptic weights
        syn_type : str
            type of synapse ("exc", "inh", None)
        tracelim : float
            clamping parameter for synaptic traces
        
        """

        super().__init__(Ne, No, W0, transform)

        self.tre = tre
        self.tro = tro
        self.Wlim = float(Wlim) if Wlim is not None else Wlim
        self.tracelim = tracelim
        self.rule_params = rule_params

        if isinstance(self.tre, Trace):
            self.te=self.tre
        else:
            self.te = Trace(self.Ne, self.tre[0], self.tre[1], self.tracelim)
        if isinstance(self.tro, Trace):
            self.to=self.tro
        else:
            self.to = Trace(self.No, self.tro[0], self.tro[1], self.tracelim)
        self.syn_type = syn_type
        # self.dWp=np.empty_like(self.W)
        # self.dWn=np.empty_like(self.W)


    def reset(self):

        super().reset()
        self.te.reset()
        self.to.reset()

    def reset_stats(self):
        self.power_forward=0
        self.energy_update=0
        self.energy_update_cell=0
        self.energy_wire_pre=0
        self.energy_wire_post=0
        self.count_input_spike=0
        self.count_output_spike=0
        self.sum_v_pre=0
        self.sum_v2_pre=0
        self.sum_v_post=0
        self.sum_v2_post=0

    def __call__(self, xe, ):

        self.xe = self.transform(xe)
        xe_index=(self.xe!=0)
        if self.syn_type == "inh":
            return - (self.W[:,xe_index]) @ (self.xe[xe_index])
        else:
            return (self.W[:,xe_index]) @ (self.xe[xe_index])



    def update(self, xo, learn=True):

        if learn:

            self.te.update(self.xe)
            self.to.update(xo)

            self.apply_rule(self.xe, xo)



    def apply_rule(self, xe, xo):
        
        xo_index=(xo!=0)
        #xo_view = self.W[xo_index,:]
        self.W[xo_index,:] += np.outer(xo[xo_index], self.rule_params["Ap"]*self.te())
        xe_index=(xe!=0)
        #xe_view = self.W[:,xe_index]
        self.W[:,xe_index] -= np.outer(self.rule_params["An"]*self.to(), xe[xe_index])
        
        self.W[xo_index,:]=np.minimum(self.W[xo_index,:], self.Wlim)
        self.W[:,xe_index]=np.maximum(self.W[:,xe_index], -self.Wlim if self.syn_type is None else 0)

class PlasticSynapseOuterproduct(PlasticSynapse):
    def apply_rule(self, xe, xo):
        
        self.W = self.W+np.outer(self.to(), self.te()*self.rule_params["lr"])

class BoundedPlasticSynapse(PlasticSynapse):
    def __init__(self, Ne, No, W0, tre, tro, transform=None,
        rule_params = None, Wlim=1, syn_type=None, tracelim=10):
        super().__init__(Ne, No, W0, tre, tro, transform, rule_params, Wlim, syn_type, tracelim)
        self.wmax=Wlim
        self.wmin=-self.Wlim if self.syn_type is None else 0
        self.enable_update=True

    def apply_rule(self, ___xe, xo):
        if self.enable_update:
            xo_index=(xo!=0)
            wrel=(self.W[xo_index,:]-self.wmin)/(self.wmax-self.wmin)
            self.W[xo_index,:]+=np.where(self.te()[np.newaxis, :]>0.5, self.rule_params["Ap"]*np.exp(-self.rule_params["Bp"]*wrel), -self.rule_params["An"]*np.exp(-self.rule_params["Bn"]*(1-wrel)))
            
            self.W[xo_index,:]=np.minimum(self.W[xo_index,:], self.wmax)
            self.W[xo_index,:]=np.maximum(self.W[xo_index,:], self.wmin)

# class PlasticConv(PlasticSynapse):
#     def __init__(self, Ce, He, We, Hp, Wp, Co, Hk, Wk, W0, tre, tro, transform=None, rule_params=None, Wlim=1, syn_type=None, tracelim=10):
#         Ho=He+Hp*2-Hk+1
#         Wo=We+Wp*2-Wk+1
#         super().__init__(self, Ce*He*We, Co*Ho*Wo, W0, tre, tro, transform, rule_params, Wlim, syn_type, tracelim)
#         self.Ce=Ce
#         self.He=He
#         self.We=We
#         self.Hp=Hp
#         self.Wp=Wp
#         self.Co=Co
#         self.Hk=Hk
#         self.Wk=Wk
#         self.Ho=Ho
#         self.Wo=Wo

class PoolScaleConvOTTT:
    def __init__(self, in_ch, in_H, in_W, out_ch, kern_size, W0, b0, te, P_kernel_size=2, P_stride=2, Xscale=2.74):
        self.W=torch.Tensor(W0.copy())
        self.b=torch.Tensor(b0.copy())
        self.P_kernel_size=P_kernel_size
        self.P_stride=P_stride
        self.Xscale=Xscale
        self.in_ch=in_ch
        self.in_H=in_H
        self.in_W=in_W
        self.te=te
    
    def __call__(self, xe):
        xe=torch.Tensor(xe).reshape(1,self.in_ch,self.in_H,self.in_W)
        if self.P_kernel_size is not None:
            xe=torchfunc.avg_pool2d(xe, kernel_size=self.P_kernel_size, stride=self.P_stride)
        xe=xe*self.Xscale
        self.te.update(xe.flatten().numpy())
        self.te_shape=xe.shape
        xe=torchfunc.conv2d(xe, self.W, self.b, padding=1)
        return xe.flatten().numpy()

    def update(self, xo, learn=True):
        pass

    def reset(self):
        self.te.reset()

class DenseOTTT:
    def __init__(self, in_N, out_N, W0, b0, te):
        self.W=W0.copy()
        self.b=b0.copy()
        self.te=te
    
    def __call__(self, xe):
        self.te.update(xe)
        return self.W@xe+self.b
    
    def update(self, xo, learn=True):
        pass

    def reset(self):
        self.te.reset()

class PoolSynapse:
    def __init__(self, w_in_raw, h_in_raw, c_in, w_stride, h_stride):
        self.w_in_raw=w_in_raw
        self.h_in_raw=h_in_raw
        self.w_in_pad=math.ceil(w_in_raw/w_stride)*w_stride
        self.h_in_pad=math.ceil(h_in_raw/h_stride)*h_stride
        self.c_in=c_in
        self.w_stride=w_stride
        self.h_stride=h_stride
        self.accumulated_in=np.zeros((self.w_in_pad, self.h_in_pad, c_in))
        self.accumulated_out=np.zeros((self.w_in_pad//w_stride, self.h_in_pad//h_stride, c_in))

    def __call__(self, xe:np.ndarray):
        x = xe.reshape((self.w_in_raw, self.h_in_raw, self.c_in))
        self.accumulated_in[:self.w_in_raw, :self.h_in_raw, :]+=x
        x = self.accumulated_in.reshape((self.w_in_pad//self.w_stride, self.w_stride, self.h_in_pad//self.h_stride, self.h_stride, self.c_in))
        x = x.max(axis=(1, 3))
        res = x-self.accumulated_out
        self.accumulated_out[:]=x
        return np.reshape(res,[-1])

    def reset(self):
        self.accumulated_in[:]=0
        self.accumulated_out[:]=0

    def update(self, pos, learn=True):
        pass


class PlasticConvSynapse(PlasticSynapse):
    def __init__(self, w_in, h_in, c_in, w_k, h_k, c_out, convmode, W0, tre, tro, transform=None, rule_params=None, Wlim=1, syn_type=None, tracelim=10):
        if convmode == 'valid':
            w_out=w_in-w_k+1
            h_out=h_in-h_k+1
            self.X=np.zeros((w_in,h_in,c_in))
        elif convmode == 'same':
            w_out=w_in
            h_out=h_in
            self.X=np.zeros((w_in+w_k-1,h_in+h_k-1,c_in))
        else:
            raise ValueError('unknown conv mode '+convmode)
        super().__init__(w_in*h_in*c_in, w_out*h_out*c_out, np.zeros(tuple()), tre, tro, transform, rule_params, Wlim, syn_type, tracelim)
        self.W=W0
        if self.W.shape != (w_k, h_k, c_in, c_out):
            raise ValueError
        self.w_in, self.h_in, self.c_in=w_in, h_in, c_in
        self.w_k, self.h_k, self.c_out=w_k, h_k, c_out
        self.w_out, self.h_out = w_out, h_out
        self.convmode=convmode
    
    def __call__(self, xe):
        self.xe=self.transform(xe)
        xe=np.reshape(self.xe,(self.w_in,self.h_in,self.c_in))
        if self.convmode == 'valid':
            self.X[:] = xe
        elif self.convmode == 'same':
            self.X[self.w_k//2:self.w_k//2+self.w_in, self.h_k//2:self.h_k//2+self.h_in, :] = xe
        
        X=extract_patches_2d(self.X, (self.w_k,self.h_k))
        if X.ndim==3:
            X=X[:,:,:,None]
        if self.syn_type == "inh":
            return np.reshape(np.reshape(X, (X.shape[0], -1)) @ np.reshape( - self.W, (-1, self.W.shape[-1])), [-1])
        else:
            return np.reshape(np.reshape(X, (X.shape[0], -1)) @ np.reshape(self.W, (-1, self.W.shape[-1])), [-1])

    def apply_rule(self, xe, xo):
        #self.W += np.outer(xo, self.rule_params["Ap"]*self.te())
        #self.W -= np.outer(self.rule_params["An"]*self.to(), xe)
        if (self.rule_params["An"]!=0 or self.rule_params["MMp"]!=0) and(xo!=0).any():
            te=np.reshape(self.te(), (self.w_in,self.h_in,self.c_in))
            if self.convmode == 'valid':
                self.X[:] = te
            elif self.convmode == 'same':
                self.X[self.w_k//2:self.w_k//2+self.w_in, self.h_k//2:self.h_k//2+self.h_in, :] = te
            X=extract_patches_2d(self.X, (self.w_out,self.h_out))
            if X.ndim==3:
                X=X[:,:,:,None]
            X=np.transpose(X, (0,3,1,2))
            outmap=np.reshape(xo, (self.w_out, self.h_out, self.c_out))
            self.W*=1-(self.rule_params["MMp"]*outmap.sum(axis=(0,1))*0.5)[None,None,None,:]
            self.W+=self.rule_params["Ap"]*np.reshape(np.reshape(X, (X.shape[0]*X.shape[1], -1)) @ np.reshape(xo, (self.w_out*self.h_out, self.c_out)), (self.w_k, self.h_k, self.c_in, self.c_out))
            self.W*=1-(self.rule_params["MMp"]*outmap.sum(axis=(0,1))*0.5)[None,None,None,:]
        
        if self.rule_params["An"]!=0 and (xe!=0).any():
            xe=np.reshape(xe, (self.w_in, self.h_in, self.c_in))
            if self.convmode == 'valid':
                self.X[:] = xe
            elif self.convmode == 'same':
                self.X[self.w_k//2:self.w_k//2+self.w_in, self.h_k//2:self.h_k//2+self.h_in, :] = xe
            X=extract_patches_2d(self.X, (self.w_out,self.h_out))
            if X.ndim==3:
                X=X[:,:,:,None]
            X=np.transpose(X, (0,3,1,2))
            self.W-=self.rule_params["An"]*np.reshape(np.reshape(X, (X.shape[0]*X.shape[1], -1)) @ np.reshape(self.to(), (self.w_out*self.h_out, self.c_out)), (self.w_k, self.h_k, self.c_in, self.c_out))

        self.W[:]=np.minimum(self.W, self.Wlim)
        self.W[:]=np.maximum(self.W, -self.Wlim if self.syn_type is None else 0)



class SynapseCircuit(PlasticSynapse):
    def __init__(self, synapse_cell, Ne, No, tre, tro, transform=None, syn_type=None, tracelim=10):
        super().__init__(Ne, No, np.ndarray(0), tre, tro, transform, None, None, syn_type, tracelim)
        self.synapse_cell=synapse_cell
        self.W=np.ones((No, Ne))
        self.synapse_cell.reset(self.W)
        self.tev_prev=None
        self.tov_prev=None
    
    def reset(self):
        super().reset()
        # self.synapse_cell.reset(self.W)
    
    def apply_rule(self, xe, xo):
        # TODO: transform required?
        te=self.te()
        to=self.to()
        tev=self.synapse_cell.te2v(te)
        tov=self.synapse_cell.to2v(to)
        # assert tev.shape==te.shape+(self.synapse_cell.Nwire_E,)
        # assert tov.shape==to.shape+(self.synapse_cell.Nwire_O,)
        # if self.tev_prev is None:
        #     self.tev_prev=np.zeros_like(tev)
        #     self.tov_prev=np.zeros_like(tov)
        # self.sum_CdV_pre+=np.max(tev-self.tev_prev,0).sum(axis=tuple(range(tev.ndim-1)))@self.wireC_pre*self.to.size
        # self.sum_CdV_post+=np.max(tov-self.tov_prev,0).sum(axis=tuple(range(tov.ndim-1)))@self.wireC_post*self.te.size
        self.sum_v_pre+=np.sum(tev)
        self.sum_v2_pre+=np.sum(tev*tev)
        self.sum_v_post+=np.sum(tov)
        self.sum_v2_post+=np.sum(tov*tov)
        self.count_input_spike+=np.sum(xe)
        self.count_output_spike+=np.sum(xo)
        xo_index=(xo!=0)
        xe_index=(xe!=0)
        self.energy_update+=np.sum(np.abs(self.W[xo_index,:]), axis=0)@(tev*tev)*self.synapse_cell.G1*self.synapse_cell.dt_xote
        self.energy_update+=np.sum(np.abs(self.W[:,xe_index]), axis=1)@(tov*tov)*self.synapse_cell.G1*self.synapse_cell.dt_xeto
        warnings.warn('Warning: energy calculation applies to 1R only')
        self.synapse_cell.apply_rule(xe, xo, self.synapse_cell.te2v(te), self.synapse_cell.to2v(to), self.W)

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, W):
        cell=self.synapse_cell
        cell.t=cell.w2t(W, cell.G1)
        self._W=cell.t2w(cell.t)
    
    def __call__(self, xe):
        result=super().__call__(xe)
        xe_index=(self.xe!=0)
        xe_sel=self.xe[xe_index]
        self.power_forward+=np.sum(np.abs(self.W[:,xe_index]), axis=0)@(xe_sel*xe_sel)*self.synapse_cell.Vread*self.synapse_cell.Vread*self.synapse_cell.G1
        return result

def get_trace_vals(tes, tos, **kwargs):
    tes_val=[]
    for te in tes:
        if isinstance(te, Trace):
            tes_val.append(te())
        elif isinstance(te, tuple) and len(te)==2:
            tes_val.append(kwargs[te[0]]*te[1])
        else:
            tes_val.append(te)
    tos_val=[]
    for to in tos:
        if isinstance(to, Trace):
            tos_val.append(to())
        elif isinstance(to, tuple) and len(to)==2:
            tos_val.append(kwargs[to[0]]*to[1])
        else:
            tos_val.append(to)
    return tes_val, tos_val

class SynapseCircuitMultiwire(PlasticSynapse):
    def __init__(self, synapse_cell, Ne, No, tres, tros, Nwire_e, Nwire_o, cap_e, cap_o, ops, op_r, transform=None, syn_type=None, tracelim=None):
        super().__init__(Ne, No, np.ndarray(0), tres, tros, transform, None, None, syn_type, tracelim)
        self.synapse_cell=synapse_cell
        self.W=np.ones((No, Ne))
        self.synapse_cell.reset(self.W)
        self.tev_prev=np.zeros((Nwire_e, Ne))
        self.tov_prev=np.zeros((Nwire_o, No))
        self.cap_e=cap_e
        self.cap_o=cap_o
        self.ops=ops
        self.op_r=op_r
        #self.energy_posts=np.zeros_like(self.cap_o)
    
    def reset(self):
        super().reset()
        # self.synapse_cell.reset(self.W)
        #self.energy_posts[:]=0
    
    def apply_rule(self, xe, xo):
        self.count_input_spike+=np.sum(xe)
        self.count_output_spike+=np.sum(xo)
        for tes, tos in self.ops:
            tes_val, tos_val = get_trace_vals(tes, tos, xe=xe, xo=xo)
            dissipated=self.synapse_cell.apply_rule([x.copy() for x in tes_val], [x.copy() for x in tos_val], self._W)
            self.energy_update_cell+=dissipated
            self._wire_state_update(tes_val, tos_val)

    def _wire_state_update(self, tes_val, tos_val):
        tev_diff=np.array(tes_val)-self.tev_prev
        self.energy_wire_pre+=np.clip(tev_diff,0,None).sum(axis=1)@self.cap_e
        tov_diff=np.array(tos_val)-self.tov_prev
        self.energy_wire_post+=np.clip(tov_diff,0,None).sum(axis=1)@self.cap_o
        #self.energy_posts+=np.clip(tov_diff,0,None).sum(axis=1)*self.cap_o
        self.tev_prev=np.array(tes_val)
        self.tov_prev=np.array(tos_val)

    @property
    def W(self):
        return self._W
    
    @W.setter
    def W(self, W):
        cell=self.synapse_cell
        cell.t=cell.w2t(W, cell.G1)
        self._W=cell.t2w(cell.t)
    
    def __call__(self, xe):
        result=super().__call__(xe)
        xe_index=(self.xe!=0)
        xe_sel=self.xe[xe_index]
        cell=self.synapse_cell
        self.power_forward+=np.sum(np.abs(self.W[:,xe_index]), axis=0)@(xe_sel*xe_sel)*cell.Vread*cell.Vread*cell.G1
        if 'incl_off' in dir(cell) and cell.incl_off:
            self.power_forward+=self.No*(cell.leak_cur(cell.Vread)*cell.Vread+cell.leak_cur(cell.vg_on)*cell.vg_on)*np.sum(xe_index)
        tes, tos = self.op_r
        tes_val, tos_val = get_trace_vals(tes,tos, xe=xe)
        self._wire_state_update(tes_val, tos_val)
        return result

class SynapseCircuitMultiwire_bp(SynapseCircuitMultiwire):
    def __init__(self, synapse_cell, Ne, No, tres, tros, Nwire_e, Nwire_o, cap_e, cap_o, ops, op_r, ops_bp, transform=None, syn_type=None, tracelim=None):
        super().__init__(synapse_cell, Ne, No, tres, tros, Nwire_e, Nwire_o, cap_e, cap_o, ops, op_r, transform, syn_type, tracelim)
        self.ops_bp=ops_bp
    
    def reset_stats(self):
        super().reset_stats()
        self.power_reverse=0
    
    def reverse(self, erro):
        errp=np.max(erro,0)
        errn=np.max(-erro,0)
        xo_index=(erro!=0)
        if self.syn_type == "inh":
            result= - (erro[xo_index]) @ (self.W[xo_index,:])
        else:
            result= (erro[xo_index]) @ (self.W[xo_index,:])
        
        xo_sel=erro[xo_index]
        self.power_reverse+=(xo_sel*xo_sel)@np.sum(np.abs(self.W[xo_index,:]), axis=1)*self.synapse_cell.Vread*self.synapse_cell.Vread*self.synapse_cell.G1
        
        for tes, tos in self.ops_bp:
            tes_val, tos_val = get_trace_vals(tes,tos, errp=errp, errn=errn)
            self._wire_state_update(tes_val, tos_val)
        return result
    
    def apply_rule(self, xe, xo):
        for tes, tos in self.ops:
            tes_val, tos_val = get_trace_vals(tes, tos)
            dissipated=self.synapse_cell.apply_rule([x.copy() for x in tes_val], [x.copy() for x in tos_val], self._W)
            self.energy_update_cell+=dissipated
            self._wire_state_update(tes_val, tos_val)

class cell_7T1R:
    def w2t(self, w, G1):
        g=w*G1
        return np.clip(g, self.gmin, self.gmax)
    
    def leak_cur(self, v):
        raise NotImplementedError
    
    def __init__(self, t0, G1, dt_xeto, dt_xote, datafile, Vread, incl_off=False, vg_on=None):
        self.Vread=Vread
        self.G1=G1
        self.t0=t0.copy()
        self.t=self.t0.copy()
        self.incl_off=incl_off
        if incl_off: # include off-state transistor
            self.vg_on=vg_on
        datafile=np.load(datafile)
        self.gmin=datafile['gmin']
        self.gmax=datafile['gmax']

        pot_G=datafile['pot_G']
        pot_V=datafile['pot_V']
        self.min_pot_v=pot_V.min()
        self.max_pot_v=pot_V.max()
        pot_dG=datafile['pot_dG']
        pot_E=datafile['pot_E']
        
        dep_G=datafile['dep_G']
        dep_V=datafile['dep_V']
        self.min_dep_v=dep_V.min()
        self.max_dep_v=dep_V.max()
        dep_dG=datafile['dep_dG']
        dep_E=datafile['dep_E']

        self.dt_xeto=dt_xeto
        self.dt_xote=dt_xote

        self.vsdepL=datafile['vs_dep_l'].astype(np.float32)
        self.vsdepH=datafile['vs_dep_h'].astype(np.float32)
        self.vspotL=datafile['vs_pot_l'].astype(np.float32)
        self.vspotH=datafile['vs_pot_h'].astype(np.float32)

        if dt_xote!=datafile['pot_dt']:
            raise ValueError('wrong pot dt')
        if dt_xeto!=datafile['dep_dt']:
            raise ValueError('wrong dep dt')

        pot_G_id=(datafile['pot_dG']!=1.0).all(axis=1)
        dep_G_id=(datafile['dep_dG']!=1.0).all(axis=1)
        self.gmin=max(self.gmin, pot_G[pot_G_id].min(), dep_G[dep_G_id].min())
        self.gmax=min(self.gmax, pot_G[pot_G_id].max(), dep_G[dep_G_id].max())
        self.potdGlut=scipy.interpolate.RegularGridInterpolator((pot_G[pot_G_id],pot_V),pot_dG[pot_G_id,:], 'linear', bounds_error=True)
        self.depdGlut=scipy.interpolate.RegularGridInterpolator((dep_G[dep_G_id],dep_V),dep_dG[dep_G_id,:], 'linear', bounds_error=True)
        self.potElut=scipy.interpolate.RegularGridInterpolator((pot_G[pot_G_id],pot_V),pot_E[pot_G_id,:], 'linear', bounds_error=True)
        self.depElut=scipy.interpolate.RegularGridInterpolator((dep_G[dep_G_id],dep_V),dep_E[dep_G_id,:], 'linear', bounds_error=True)

    def t2w(self, t):
        return t/self.G1

    def reset(self, wout):
        self.t[:]=self.t0
        wout[:]=self.t2w(self.t)
    
    def apply_rule(self, tes_val, tos_val, Wout):
        vpre, vsr, vsdepL, vsdepH = tes_val
        vpost, vspotL, vspotH = tos_val
        if np.unique(vsr).tolist()!=[0.0]:
            raise ValueError('read not allowed')
        if not set(np.unique(vsdepL).tolist()).issubset({0.0, self.vsdepL[()]}):
            raise ValueError('illegal vsdepL')
        if not set(np.unique(vsdepH).tolist()).issubset({0.0, self.vsdepH[()]}):
            raise ValueError('illegal vsdepH')
        if not set(np.unique(vspotL).tolist()).issubset({0.0, self.vspotL[()]}):
            raise ValueError('illegal vspotL')
        if not set(np.unique(vspotH).tolist()).issubset({0.0, self.vspotH[()]}):
            raise ValueError('illegal vspotH')
        if ((vsdepL!=0)!=(vsdepH!=0)).any():
            raise ValueError('mismatch vsdep[L,H]')
        if ((vspotL!=0)!=(vspotH!=0)).any():
            raise ValueError('mismatch vspot[L,H]')
        xe_index=(vsdepL!=0)
        xo_index=(vspotL!=0)

        dissipated=0
        if xe_index.any():
            # dep mode
            if xo_index.any():
                raise ValueError('dep and pot not allowed')
            to_index=(vpost>=self.min_dep_v)
            if to_index.any():
                vpost[vpost>self.max_dep_v]=self.max_dep_v
                vpost[np.logical_not(to_index)]=self.min_dep_v
                dissipated+=np.where(to_index[:,np.newaxis],self.depElut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vpost[:,np.newaxis]), axis=-1), 'linear'),0).sum()
                self.t[:,xe_index]+=np.where(to_index[:,np.newaxis],self.depdGlut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vpost[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gmin, self.gmax)
                Wout[:,xe_index]=self.t2w(self.t[:,xe_index])
        elif xo_index.any():
            # pot mode
            te_index=(vpre>=self.min_pot_v)
            if te_index.any():
                vpre[vpre>self.max_pot_v]=self.max_pot_v
                vpre[np.logical_not(te_index)]=self.min_pot_v
                dissipated+=np.where(te_index[np.newaxis,:],self.potElut(np.stack(np.broadcast_arrays(self.t[xo_index,:], vpre[np.newaxis,:]), axis=-1),'linear'),0).sum()
                self.t[xo_index,:]+=np.where(te_index[np.newaxis,:],self.potdGlut(np.stack(np.broadcast_arrays(self.t[xo_index,:], vpre[np.newaxis,:]), axis=-1),'linear'),0)
                self.t[xo_index]=np.clip(self.t[xo_index], self.gmin, self.gmax)
                Wout[xo_index]=self.t2w(self.t[xo_index])
        return np.abs(dissipated)

class cell_1T1R:
    def w2t(self, w, G1):
        g=w*G1
        return np.clip(g, self.gmin, self.gmax)
    
    def __init__(self, t0, G1, dt_xeto, dt_xote, datafile, Vread):
        self.Vread=Vread
        self.G1=G1
        self.t0=t0.copy()
        self.t=self.t0.copy()
        datafile=np.load(datafile)
        self.gmin=datafile['gmin']
        self.gmax=datafile['gmax']

        pot_G=datafile['pot_G']
        pot_V=datafile['pot_V']
        self.min_pot_v=pot_V.min()
        self.max_pot_v=pot_V.max()
        pot_dG=datafile['pot_dG']
        pot_E=datafile['pot_E']
        
        dep_G=datafile['dep_G']
        dep_V=datafile['dep_V']
        self.min_dep_v=dep_V.min()
        self.max_dep_v=dep_V.max()
        dep_dG=datafile['dep_dG']
        dep_E=datafile['dep_E']

        self.dt_xeto=dt_xeto
        self.dt_xote=dt_xote

        self.vpot_TE=datafile['vpot_TE'].astype(np.float32)
        self.vdep_G=datafile['vdep_G'].astype(np.float32)

        if dt_xote!=datafile['pot_dt']:
            raise ValueError('wrong pot dt')
        if dt_xeto!=datafile['dep_dt']:
            raise ValueError('wrong dep dt')

        pot_G_id=(datafile['pot_dG']!=1.0).all(axis=1)
        dep_G_id=(datafile['dep_dG']!=1.0).all(axis=1)
        self.gmin=max(self.gmin, pot_G[pot_G_id].min(), dep_G[dep_G_id].min())
        self.gmax=min(self.gmax, pot_G[pot_G_id].max(), dep_G[dep_G_id].max())
        self.potdGlut=scipy.interpolate.RegularGridInterpolator((pot_G[pot_G_id],pot_V),pot_dG[pot_G_id,:], 'linear', bounds_error=True)
        self.depdGlut=scipy.interpolate.RegularGridInterpolator((dep_G[dep_G_id],dep_V),dep_dG[dep_G_id,:], 'linear', bounds_error=True)
        self.potElut=scipy.interpolate.RegularGridInterpolator((pot_G[pot_G_id],pot_V),pot_E[pot_G_id,:], 'linear', bounds_error=True)
        self.depElut=scipy.interpolate.RegularGridInterpolator((dep_G[dep_G_id],dep_V),dep_E[dep_G_id,:], 'linear', bounds_error=True)

    def t2w(self, t):
        return t/self.G1

    def reset(self, wout):
        self.t[:]=self.t0
        wout[:]=self.t2w(self.t)
    
    def apply_rule(self, tes_val, tos_val, Wout):
        vg, = tes_val
        vte, = tos_val

        xe_index=(vg!=0)
        xo_index=(vte!=0)

        dissipated=0
        if set(np.unique(vg).tolist()).issubset({0.0, self.vdep_G[()]}):
          if xe_index.any():
            # dep mode
            to_index=(vte>=self.min_dep_v)
            if to_index.any():
                vte[vte>self.max_dep_v]=self.max_dep_v
                vte[np.logical_not(to_index)]=self.min_dep_v
                dissipated+=np.where(to_index[:,np.newaxis],self.depElut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vte[:,np.newaxis]), axis=-1), 'linear'),0).sum()
                self.t[:,xe_index]+=np.where(to_index[:,np.newaxis],self.depdGlut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vte[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gmin, self.gmax)
                Wout[:,xe_index]=self.t2w(self.t[:,xe_index])
        elif set(np.unique(vte).tolist()).issubset({0.0, self.vpot_TE[()]}):
          if xo_index.any():
            # pot mode
            te_index=(vg>=self.min_pot_v)
            if te_index.any():
                vg[vg>self.max_pot_v]=self.max_pot_v
                vg[np.logical_not(te_index)]=self.min_pot_v
                dissipated+=np.where(te_index[np.newaxis,:],self.potElut(np.stack(np.broadcast_arrays(self.t[xo_index,:], vg[np.newaxis,:]), axis=-1),'linear'),0).sum()
                self.t[xo_index,:]+=np.where(te_index[np.newaxis,:],self.potdGlut(np.stack(np.broadcast_arrays(self.t[xo_index,:], vg[np.newaxis,:]), axis=-1),'linear'),0)
                self.t[xo_index]=np.clip(self.t[xo_index], self.gmin, self.gmax)
                Wout[xo_index]=self.t2w(self.t[xo_index])
        else:
            raise ValueError('unrecognizable VG and VTE')
        return np.abs(dissipated)

class cell_7T1R_bp:
    def w2t(self, w, G1):
        g=w*G1
        return np.clip(g, self.gmin, self.gmax)
    
    def __init__(self, t0, G1, dt_xeto, datafile, Vread, incl_off=False, vg_on=None):
        self.Vread=Vread
        self.G1=G1
        self.t0=t0.copy()
        self.t=self.t0.copy()
        self.incl_off=incl_off
        if incl_off: # include off-state transistor
            self.vg_on=vg_on
        datafile=np.load(datafile)
        self.gmin=datafile['gmin']
        self.gmax=datafile['gmax']

        # pot_G=datafile['pot_G']
        # pot_V=datafile['pot_V']
        # self.min_pot_v=pot_V.min()
        # self.max_pot_v=pot_V.max()
        # pot_dG=datafile['pot_dG']
        # pot_E=datafile['pot_E']
        
        dep_G=datafile['dep_G']
        dep_V=datafile['dep_V']
        self.min_dep_v=dep_V.min()
        self.max_dep_v=dep_V.max()
        dep_dG=datafile['dep_dG']
        dep_E=datafile['dep_E']

        self.dt_xeto=dt_xeto
        # self.dt_xote=dt_xote

        self.vsdepL=datafile['vs_dep_l'].astype(np.float32)
        self.vsdepH=datafile['vs_dep_h'].astype(np.float32)
        # self.vspotL=datafile['vs_pot_l'].astype(np.float32)
        # self.vspotH=datafile['vs_pot_h'].astype(np.float32)

        # if dt_xote!=datafile['pot_dt']:
        #     raise ValueError('wrong pot dt')
        if dt_xeto!=datafile['dep_dt']:
            raise ValueError('wrong dep dt')

        # pot_G_id=(datafile['pot_dG']!=1.0).all(axis=1)
        dep_G_id=(datafile['dep_dG']!=1.0).all(axis=1)
        # self.gmin=max(self.gmin, pot_G[pot_G_id].min(), dep_G[dep_G_id].min())
        # self.gmax=min(self.gmax, pot_G[pot_G_id].max(), dep_G[dep_G_id].max())
        self.gmin=max(self.gmin, dep_G[dep_G_id].min())
        self.gmax=min(self.gmax, dep_G[dep_G_id].max())
        # self.potdGlut=scipy.interpolate.RegularGridInterpolator((pot_G[pot_G_id],pot_V),pot_dG[pot_G_id,:], 'linear', bounds_error=True)
        self.depdGlut=scipy.interpolate.RegularGridInterpolator((dep_G[dep_G_id],dep_V),dep_dG[dep_G_id,:], 'linear', bounds_error=True)
        # self.potElut=scipy.interpolate.RegularGridInterpolator((pot_G[pot_G_id],pot_V),pot_E[pot_G_id,:], 'linear', bounds_error=True)
        self.depElut=scipy.interpolate.RegularGridInterpolator((dep_G[dep_G_id],dep_V),dep_E[dep_G_id,:], 'linear', bounds_error=True)

    def t2w(self, t):
        return t/self.G1

    def reset(self, wout):
        self.t[:]=self.t0
        wout[:]=self.t2w(self.t)
    
    def apply_rule(self, tes_val, tos_val, Wout):
        vsel, vserh, vseph, vsenh = tes_val
        vsol, vsoh, verrp, verrn = tos_val
        if np.unique(vserh).tolist()!=[0.0]:
            raise ValueError('read not allowed')
        if (vsol!=0).any() or (vsoh!=0).any():
            raise ValueError('reverse read not allowed')
        if not set(np.unique(vsel).tolist()).issubset({0.0, self.vsdepL[()]}):
            raise ValueError('illegal vsel')
        if not set(np.unique(vseph).tolist()).issubset({0.0, self.vsdepH[()]}):
            raise ValueError('illegal vseph')
        if not set(np.unique(vsenh).tolist()).issubset({0.0, self.vsdepH[()]}):
            raise ValueError('illegal vsenh')
        if not set(np.unique(vsol).tolist()).issubset({0.0, self.vsdepL[()]}):
            raise ValueError('illegal vsol')
        if not set(np.unique(vsoh).tolist()).issubset({0.0, self.vsdepH[()]}):
            raise ValueError('illegal vsoh')
        if (verrp<0).any() or (verrn<0).any():
            raise ValueError('verrp/n neg')
        if ((vsel!=0)!=np.logical_or(vseph!=0, vsenh!=0)).any():
            raise ValueError('mismatch vse[L,H]')
        xe_index_p=(vseph!=0)
        xe_index_n=(vsenh!=0)
        
        dissipated=0
        if xe_index_p.any():
            # pos mode
            if xe_index_n.any():
                raise ValueError('dep and pot not allowed')
            to_index=(verrp>=self.min_dep_v)
            if to_index.any():
                if verrp.max()>self.max_dep_v+1/30:
                    warnings.warn(f'verrp too large: {verrp.max()} while max_dep_v is {self.max_dep_v}')
                verrp[verrp>self.max_dep_v]=self.max_dep_v
                verrp[np.logical_not(to_index)]=self.min_dep_v
                dissipated+=np.where(to_index[:,np.newaxis],self.depElut(np.stack(np.broadcast_arrays(self.t[:,xe_index_p], verrp[:,np.newaxis]), axis=-1), 'linear'),0).sum()
                self.t[:,xe_index_p]-=np.where(to_index[:,np.newaxis],self.depdGlut(np.stack(np.broadcast_arrays(self.t[:,xe_index_p], verrp[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index_p]=np.clip(self.t[:,xe_index_p], self.gmin, self.gmax)
                Wout[:,xe_index_p]=self.t2w(self.t[:,xe_index_p])
        elif xe_index_n.any():
            # neg mode
            to_index=(verrn>=self.min_dep_v)
            if to_index.any():
                if verrn.max()>self.max_dep_v+1/30:
                    warnings.warn(f'verrn too large: {verrn.max()} while max_dep_v is {self.max_dep_v}')
                verrn[verrn>self.max_dep_v]=self.max_dep_v
                verrn[np.logical_not(to_index)]=self.min_dep_v
                dissipated+=np.where(to_index[:,np.newaxis],self.depElut(np.stack(np.broadcast_arrays(self.t[:,xe_index_n], verrn[:,np.newaxis]), axis=-1), 'linear'),0).sum()
                self.t[:,xe_index_n]+=np.where(to_index[:,np.newaxis],self.depdGlut(np.stack(np.broadcast_arrays(self.t[:,xe_index_n], verrn[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index_n]=np.clip(self.t[:,xe_index_n], self.gmin, self.gmax)
                Wout[:,xe_index_n]=self.t2w(self.t[:,xe_index_n])
        return np.abs(dissipated) # HSPICE energy numbers are all negative, NAN is +1.0 in ...Elut

class cell_1T1R_bp:
    def w2t(self, w, G1):
        g=w*G1
        return np.clip(g, self.gmin, self.gmax)
    
    def __init__(self, t0, G1, dt_xeto, datafile, Vread):
        self.Vread=Vread
        self.G1=G1
        self.t0=t0.copy()
        self.t=self.t0.copy()
        datafile=np.load(datafile)
        self.gmin=datafile['gmin']
        self.gmax=datafile['gmax']

        # pot_G=datafile['pot_G']
        # pot_V=datafile['pot_V']
        # self.min_pot_v=pot_V.min()
        # self.max_pot_v=pot_V.max()
        # pot_dG=datafile['pot_dG']
        # pot_E=datafile['pot_E']
        
        dep_G=datafile['dep_G']
        dep_V=datafile['dep_V']
        self.min_dep_v=dep_V.min()
        self.max_dep_v=dep_V.max()
        dep_dG=datafile['dep_dG']
        dep_E=datafile['dep_E']

        self.dt_xeto=dt_xeto
        # self.dt_xote=dt_xote

        # self.vpot_TE=datafile['vpot_TE'].astype(np.float32)
        self.vdep_G=datafile['vdep_G'].astype(np.float32)

        # if dt_xote!=datafile['pot_dt']:
        #     raise ValueError('wrong pot dt')
        if dt_xeto!=datafile['dep_dt']:
            raise ValueError('wrong dep dt')

        # pot_G_id=(datafile['pot_dG']!=1.0).all(axis=1)
        dep_G_id=(datafile['dep_dG']!=1.0).all(axis=1)
        # self.gmin=max(self.gmin, pot_G[pot_G_id].min(), dep_G[dep_G_id].min())
        # self.gmax=min(self.gmax, pot_G[pot_G_id].max(), dep_G[dep_G_id].max())
        self.gmin=max(self.gmin, dep_G[dep_G_id].min())
        self.gmax=min(self.gmax, dep_G[dep_G_id].max())
        # self.potdGlut=scipy.interpolate.RegularGridInterpolator((pot_G[pot_G_id],pot_V),pot_dG[pot_G_id,:], 'linear', bounds_error=True)
        self.depdGlut=scipy.interpolate.RegularGridInterpolator((dep_G[dep_G_id],dep_V),dep_dG[dep_G_id,:], 'linear', bounds_error=True)
        # self.potElut=scipy.interpolate.RegularGridInterpolator((pot_G[pot_G_id],pot_V),pot_E[pot_G_id,:], 'linear', bounds_error=True)
        self.depElut=scipy.interpolate.RegularGridInterpolator((dep_G[dep_G_id],dep_V),dep_E[dep_G_id,:], 'linear', bounds_error=True)

    def t2w(self, t):
        return t/self.G1

    def reset(self, wout):
        self.t[:]=self.t0
        wout[:]=self.t2w(self.t)
    
    def apply_rule(self, tes_val, tos_val, Wout):
        vg, = tes_val
        vte, = tos_val
        if not set(np.unique(vg).tolist()).issubset({0.0, self.vdep_G[()]}):
            raise ValueError('gate require digital')
        if (vte>0).any() and (vte<0).any():
            raise ValueError('err pos and neg mix')
        xe_index=(vg!=0)

        dissipated=0
        if (vte>0).any():
          if xe_index.any():
            # pos mode
            to_index=(vte>=self.min_dep_v)
            if to_index.any():
                if vte.max()>self.max_dep_v+1/30:
                    warnings.warn(f'vte too large: {vte.max()} while max_dep_v is {self.max_dep_v}')
                vte[vte>self.max_dep_v]=self.max_dep_v
                vte[np.logical_not(to_index)]=self.min_dep_v
                dissipated+=np.where(to_index[:,np.newaxis],self.depElut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vte[:,np.newaxis]), axis=-1), 'linear'),0).sum()
                self.t[:,xe_index]-=np.where(to_index[:,np.newaxis],self.depdGlut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vte[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gmin, self.gmax)
                Wout[:,xe_index]=self.t2w(self.t[:,xe_index])
        elif (vte<0).any():
          if xe_index.any():
            # neg mode
            vte[:]=-vte
            to_index=(vte>=self.min_dep_v)
            if to_index.any():
                if vte.max()>self.max_dep_v+1/30:
                    warnings.warn(f'vte too large: {vte.max()} while max_dep_v is {self.max_dep_v}')
                vte[vte>self.max_dep_v]=self.max_dep_v
                vte[np.logical_not(to_index)]=self.min_dep_v
                dissipated+=np.where(to_index[:,np.newaxis],self.depElut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vte[:,np.newaxis]), axis=-1), 'linear'),0).sum()
                self.t[:,xe_index]+=np.where(to_index[:,np.newaxis],self.depdGlut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vte[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gmin, self.gmax)
                Wout[:,xe_index]=self.t2w(self.t[:,xe_index])
        return np.abs(dissipated)

class SynapseCircuitBP(SynapseCircuit):
    def apply_rule(self, xe, xo):
        # xe may not be binary
        to=self.to()
        to_p=np.clip(to, 0, None)
        to_n=-np.clip(to, None, 0)
        tov_p=self.synapse_cell.te2v(to_p)
        tov_n=self.synapse_cell.to2v(to_n)

        self.sum_v_pre+=np.sum(tov_p)
        self.sum_v2_pre+=np.sum(tov_p*tov_p)
        self.sum_v_post+=np.sum(tov_n)
        self.sum_v2_post+=np.sum(tov_n*tov_n)
        self.count_input_spike+=np.sum(np.clip(xe,None,1))*2
        xe_index=(xe!=0)
        self.energy_update+=np.abs(self.W[:,xe_index])@(xe[xe_index])@(tov_p*tov_p)*self.synapse_cell.G1*self.synapse_cell.dt_xeto
        self.energy_update+=np.abs(self.W[:,xe_index])@(xe[xe_index])@(tov_n*tov_n)*self.synapse_cell.G1*self.synapse_cell.dt_xeto
        warnings.warn('Warning: energy calculation applies to 1R only')

        for th in range(xe.max()):
            self.synapse_cell.apply_rule(xe>th, np.zeros_like(xo), np.zeros_like(xe), tov_p, self.W)
        for th in range(xe.max()):
            self.synapse_cell.apply_rule(xe>th, np.zeros_like(xo), np.zeros_like(xe), tov_n, self.W)

class cell_Paiyu_chen_15:
    g0=2.7505e-10
    I0=6.14e-5
    Vread=0.7
    V0=0.43
    I0_VrshVr_v0=I0/Vread*math.sinh(Vread/V0)
    gamma0=16.5
    beta=1.25
    g1=1e-9
    Vel0=150
    q=1.60217663e-19
    Eag=1.501
    Ear=1.5
    kb=1.3806503e-23
    temperature=350
    a0=0.25e-9
    L=5e-9
    eqEagkt=math.exp(-q*Eag/kb/temperature)
    eqEarkt=math.exp(-q*Ear/kb/temperature)
    alqkt=a0/L*q/kb/temperature
    gap_min=0.1e-9
    gap_max=1.7e-9

    @classmethod
    def w2t(cls, w, G1):
        t=-np.log(w/cls.I0_VrshVr_v0*G1)*cls.g0
        return np.clip(t, cls.gap_min, cls.gap_max)

    def __init__(self, t0, G1, te2v, to2v, dt_xeto, dt_xote, nsubstep):
        self.t0=t0.copy()
        self.t=self.t0.copy()
        self.G1=G1
        self.te2v=te2v
        self.to2v=to2v
        self.dt_xeto_each=dt_xeto/nsubstep
        self.dt_xote_each=dt_xote/nsubstep
        self.dt_xeto=dt_xeto
        self.dt_xote=dt_xote
        self.nsubstep=nsubstep

    def t2w(self, t):
        return self.I0_VrshVr_v0/self.G1*np.exp(-t/self.g0)

    def reset(self, wout):
        self.t[:]=self.t0
        wout[:]=self.t2w(self.t)

    def apply_rule(self, xe, xo, ve, vo, wout):
        # ve=self.te2v(te)
        xo_index=(xo!=0)
        # gap_collect=np.zeros((self.nsubstep, xo[xo_index].size, te.size))
        if xo_index.any():
          for i in range(self.nsubstep):
            gamma = self.gamma0 - self.beta*np.power(self.t[xo_index,:]/self.g1, 3)
            gap_ddt = -self.Vel0*( self.eqEagkt*np.exp(gamma*ve[np.newaxis,:]*self.alqkt) - self.eqEarkt*np.exp(-gamma*ve[np.newaxis,:]*self.alqkt) )
            self.t[xo_index,:]+=gap_ddt*self.dt_xote_each
            # gap_collect[i,:]=np.abs(gap_ddt)
        self.t[xo_index,:]=np.clip(self.t[xo_index,:], self.gap_min, self.gap_max)
        # if gap_collect.size>0:
        #     diff=(gap_collect.max(axis=0)/gap_collect.min(axis=0)).max()
        #     if diff>1.02:
        #         print('xo:', diff)
        wout[xo_index,:]=self.t2w(self.t[xo_index,:])

        # vo=self.to2v(to)
        xe_index=(xe!=0)
        # gap_collect=np.zeros((self.nsubstep, to.size, xe[xe_index].size))
        if xe_index.any():
          for i in range(self.nsubstep):
            gamma = self.gamma0 - self.beta*np.power(self.t[:,xe_index]/self.g1, 3)
            gap_ddt = -self.Vel0*( self.eqEagkt*np.exp(gamma*vo[:,np.newaxis]*self.alqkt) - self.eqEarkt*np.exp(-gamma*vo[:,np.newaxis]*self.alqkt) )
            self.t[:,xe_index]+=gap_ddt*self.dt_xeto_each
            # gap_collect[i,:]=np.abs(gap_ddt)
        self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gap_min, self.gap_max)
        # if gap_collect.size>0:
        #     diff=(gap_collect.max(axis=0)/gap_collect.min(axis=0)).max()
        #     if diff>1.02:
        #         print('xe:', diff)
        wout[:,xe_index]=self.t2w(self.t[:,xe_index])

class cell_Vteam:
    Roff=2250521.504002251
    Ron=6697.893598309347
    Vread=0.7

    @classmethod
    def w2t(cls, w, G1):
        t=1/(w*G1)
        return np.clip(t, cls.Ron, cls.Roff)

    def __init__(self, R0, G1, te2dR, to2dR, dt_xeto, dt_xote):
        self.t0=R0.copy()
        self.t=self.t0.copy()
        self.G1=G1
        self.te2dR=te2dR
        self.to2dR=to2dR
        self.dt_xeto=dt_xeto
        self.dt_xote=dt_xote

    def te2v(self, te):
        dR=self.te2dR(te)
        assert (dR<=0).all()
        return np.power(-dR*self.G1/0.0000000528,1/41.95)

    def to2v(self, to):
        dR=self.to2dR(to)
        assert (dR>=0).all()
        return -np.power(dR*self.G1/0.0000009545,1/39.65)

    def t2w(self, t):
        return 1/self.G1/t

    def reset(self, wout):
        self.t[:]=self.t0
        wout[:]=self.t2w(self.t)

    def apply_rule(self, xe, xo, ve, vo, wout):
        # dR=self.te2dR(te)
        dR=np.where(ve>0,-np.power(ve,41.95)/self.G1*0.0000000528,np.power(-ve,39.65)/self.G1*0.0000009545)
        xo_index=(xo!=0)
        if xo_index.any():
            self.t[xo_index,:]+=dR[np.newaxis,:]
            self.t[xo_index,:]=np.clip(self.t[xo_index,:], self.Ron, self.Roff)
            wout[xo_index,:]=self.t2w(self.t[xo_index,:])
        
        # dR=self.to2dR(to)
        dR=np.where(vo>0,-np.power(vo,41.95)/self.G1*0.0000000528,np.power(-vo,39.65)/self.G1*0.0000009545)
        xe_index=(xe!=0)
        if xe_index.any():
            self.t[:,xe_index]+=dR[:,np.newaxis]
            self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.Ron, self.Roff)
            wout[:,xe_index]=self.t2w(self.t[:,xe_index])

class cell_data_driven:
    def w2t(self, w, G1):
        g=w*G1
        return np.clip(g, self.gmin, self.gmax)
    
    def __init__(self, t0, G1, te2v, to2v, dt_xeto, dt_xote, datafile, Vread):
        self.Vread=Vread
        self.te2v=te2v
        self.to2v=to2v
        self.G1=G1
        self.t0=t0.copy()
        self.t=self.t0.copy()
        self.dt_xeto=dt_xeto
        self.dt_xote=dt_xote
        datafile=np.load(datafile)
        self.gmin=datafile['gmin']
        self.gmax=datafile['gmax']
        pot_G=datafile['pot_G']
        pot_V=datafile['pot_V']
        self.min_pot_v=pot_V.min()
        self.max_pot_v=pot_V.max()
        pot_dG=datafile['pot_dG']
        dep_G=datafile['dep_G']
        dep_V=datafile['dep_V']
        self.min_dep_v=dep_V.min()
        self.max_dep_v=dep_V.max()
        dep_dG=datafile['dep_dG']
        self.data_pot_dt=datafile['pot_dt']
        self.data_dep_dt=datafile['dep_dt']
        self.potlut=scipy.interpolate.RegularGridInterpolator((pot_G,pot_V),pot_dG, 'linear', bounds_error=True)
        self.deplut=scipy.interpolate.RegularGridInterpolator((dep_G,dep_V),dep_dG, 'linear', bounds_error=True)

    def t2w(self, t):
        return t/self.G1

    def reset(self, wout):
        self.t[:]=self.t0
        wout[:]=self.t2w(self.t)

    def apply_rule(self, xe, xo, ve, vo, wout):
        xo_index=(xo!=0)
        if xo_index.any():
            # ve=self.te2v(te)
            te_index=(ve>=self.min_pot_v)
            if te_index.any():
                #ve[ve>self.max_pot_v]=self.max_pot_v
                ve[np.logical_not(te_index)]=self.min_pot_v
                self.t[xo_index,:]+=(self.dt_xote/self.data_pot_dt)*np.where(te_index[np.newaxis,:],self.potlut(np.stack(np.broadcast_arrays(self.t[xo_index,:], ve[np.newaxis,:]), axis=-1),'linear'),0)
                self.t[xo_index]=np.clip(self.t[xo_index], self.gmin, self.gmax)
                wout[xo_index]=self.t2w(self.t[xo_index])

        xe_index=(xe!=0)
        if xe_index.any():
            # vo=self.to2v(to)
            to_index=(vo>=self.min_dep_v)
            if to_index.any():
                #vo[vo>self.max_dep_v]=self.max_dep_v
                vo[np.logical_not(to_index)]=self.min_dep_v
                self.t[:,xe_index]+=(self.dt_xeto/self.data_dep_dt)*np.where(to_index[:,np.newaxis],self.deplut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vo[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gmin, self.gmax)
                wout[:,xe_index]=self.t2w(self.t[:,xe_index])

class cell_data_driven_bp(cell_data_driven):
    def apply_rule(self, xe, xo, _ve, vo, wout):
        xo_index=(xo!=0)
        assert not xo_index.any()

        xe_index=(xe!=0)
        if xe_index.any():
            # vo=self.to2v(to)
            vo_copy=vo.copy()
            to_index=(vo>=self.min_dep_v)
            if to_index.any():
                #vo[vo>self.max_dep_v]=self.max_dep_v
                vo[np.logical_not(to_index)]=self.min_dep_v
                self.t[:,xe_index]+=(self.dt_xeto/self.data_dep_dt)*np.where(to_index[:,np.newaxis],self.deplut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vo[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gmin, self.gmax)
                wout[:,xe_index]=self.t2w(self.t[:,xe_index])
            
            vo=-vo_copy
            to_index=(vo>=self.min_dep_v)
            if to_index.any():
                #vo[vo>self.max_dep_v]=self.max_dep_v
                vo[np.logical_not(to_index)]=self.min_dep_v
                self.t[:,xe_index]-=(self.dt_xeto/self.data_dep_dt)*np.where(to_index[:,np.newaxis],self.deplut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vo[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gmin, self.gmax)
                wout[:,xe_index]=self.t2w(self.t[:,xe_index])

class cell_data_driven_bp_T(cell_data_driven):
    def apply_rule(self, xe, xo, _ve, vo, wout):
        assert not (xo!=0).any()

        xe_index=(xe!=0)
        if xe_index.any():
            # vo=self.to2v(to)
            vo_copy=vo.copy()
            vo=-vo
            to_index=(vo>=self.min_dep_v)
            if to_index.any():
                #vo[vo>self.max_dep_v]=self.max_dep_v
                vo[np.logical_not(to_index)]=self.min_dep_v
                vo[vo>self.max_dep_v]=self.max_dep_v
                self.t[:,xe_index]+=(self.dt_xeto/self.data_dep_dt)*np.where(to_index[:,np.newaxis],self.deplut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vo[:,np.newaxis]), axis=-1), 'linear'),0)
                self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gmin, self.gmax)
                wout[:,xe_index]=self.t2w(self.t[:,xe_index])
            
            vo=vo_copy
            to_index=(vo>=self.min_pot_v)
            if to_index.any():
                #ve[ve>self.max_pot_v]=self.max_pot_v
                vo[np.logical_not(to_index)]=self.min_pot_v
                vo[vo>self.max_pot_v]=self.max_pot_v
                self.t[:,xe_index]+=(self.dt_xote/self.data_pot_dt)*np.where(to_index[:,np.newaxis],self.potlut(np.stack(np.broadcast_arrays(self.t[:,xe_index], vo[:,np.newaxis]), axis=-1),'linear'),0)
                self.t[:,xe_index]=np.clip(self.t[:,xe_index], self.gmin, self.gmax)
                wout[:,xe_index]=self.t2w(self.t[:,xe_index])
