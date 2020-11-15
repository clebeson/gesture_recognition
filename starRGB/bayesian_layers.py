import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
from torch.autograd import Variable
from torch.distributions import Normal
from torch.nn.utils.rnn import PackedSequence
from enum import Enum
from scipy.linalg import block_diag

class ModelType(Enum):
    DET = 0
    BBB = 1
    VAR_DROP_A_ADAP = 2
    VAR_DROP_B_ADAP = 3
    VAR_DROP_A_FIX = 4
    VAR_DROP_B_FIX = 5
    MC_DROP = 6
    


class GMM(object):
        def __init__(self, mu1, mu2, sigma1, sigma2, pi):
            self.N1 = Normal(mu1, sigma1)
            self.N2 = Normal(mu2, sigma2)

            self.pi1 = pi
            self.pi2 = (1. - pi)

        def log_prob(self, x):
            ll1 = self.N1.log_prob(x)
            ll2 = self.N2.log_prob(x)
            ll = ll1 + ( self.pi1 + self.pi2*((ll2-ll1).exp()) ).log()
            return ll

class GaussianDistribution(object):
    def __init__(self, mu, logvar):
        self.mu = mu
        self._logvar = logvar
        self._last_sample = mu
        self._cum_kl_div = 0.0
        self._prior = None
        self._total_samples = 0.0

    @property
    def var(self):
        return self._logvar.exp()
    
    @property
    def sigma(self):
        return (0.5*self._logvar).exp()
    
    def set_prior_params(self,mu,logstd1,logstd2,pi):
        sigma1, sigma2 = math.exp(logstd1), math.exp(logstd2)
        #self._prior = GMM(mu,mu,sigma1,sigma2,pi)
        self._prior = Normal(mu,sigma1)

    def clear_samples(self):
         self._cum_kl_div = 0.0
         self._total_samples = 0.0

    #reparametrization trick
    def sample(self, store):
        # prinprint(store)
        eps = torch.randn_like(self.mu)
        self._last_sample = self.mu + self.sigma * eps
        if store:               # log posterior  - log prior
            self._cum_kl_div += (self.log_prob(self._last_sample).mean() - self._prior.log_prob(self._last_sample)).mean()
            self._total_samples += 1.0
        return self._last_sample


    def log_prob(self, sample):
        # cte = math.log(2 * math.pi)
        # inner = ((sample - self.mu) ** 2)/self.var
        # return -(0.5*inner - cte -  self._logvar )
        
        log_scale = 0.5*self._logvar
        return -((sample - self.mu) ** 2) / (2 * self.var) - log_scale - math.log(math.sqrt(2 * math.pi))


        # return -0.5*(math.log(2*math.pi)
        #             +self._logvar #log(sigma)
        #             +((sample - self.mu ) ** 2) / self.var) #esampleponencial term



    def kl_div(self):
        #return (self._cum_kl_div / (self._total_samples+1e-8)).mean()
        return self._cum_kl_div 
        # kl = self._cum_kl_div
        # Nsample = 100 - self._total_samples if self._total_samples < 100 else self._total_samples
        # sigma = self.sigma
        # for _ in range(int(Nsample)):
        #     eps = torch.randn_like(self.mu)
        #     sample = self.mu +  sigma * eps
        #     kl += (self.log_prob(sample) - self._prior.log_prob(sample)).sum()
        # return kl / float(self._total_samples+Nsample)

class Parameters(nn.Module):
    """
    Implements the base code for both bayesian and deterministic parameters
    :param size_w: weight size
    :param size_b: bias size
    :param mu: mean of the prior distribuition (only for bayesian layers)
    :param logstd: logarithm of standard deviation of the prior distribuition (only for bayesian layers)
    :param type: choose the type of the layer. It can be:
                 "BBB" - for Bayesian By Backprop (default);
                 "VARDROP" - for Bayesian Variational Dropout;
                 "BAYESDROP" - for Bayesian with Dropout; and
                 "DET" - for Deterministic
    """
  
    def __init__(self,size_w, size_b,  mu = 0, logstd1 = -1, logstd2 = -2, pi = 0.5, type = ModelType.BBB ):
        super(Parameters, self).__init__()

        self._type = type
        self._w_mu = nn.Parameter(torch.Tensor(*size_w))
        self._b_mu = nn.Parameter(torch.Tensor(size_b))
        self._values = [self._w_mu, self._b_mu]
        self._sampling = False
        self._store_samples = False
        if self._type == ModelType.BBB:
            self._w_logvar = nn.Parameter(torch.Tensor(*size_w))
            self._b_logvar = nn.Parameter(torch.Tensor(size_b))
            self._weight = GaussianDistribution(self._w_mu, self._w_logvar)
            self._bias = GaussianDistribution(self._b_mu, self._b_logvar)
            self._weight.set_prior_params(mu,logstd1,logstd2,pi)
            self._bias.set_prior_params(mu,logstd1,logstd2,pi)
            self._values += [self._w_logvar,self._b_logvar]
            self._sampling = True
            self._store_samples = True
            

        elif self._type.value in [2,3]: #VARDROP ADAPTATIVE
            self._w_logvar = nn.Parameter(torch.Tensor(*size_w))
            self._weight = self._w_mu
            self._b_logvar = nn.Parameter(torch.Tensor(size_b))
            self._bias = self._b_mu
            self._values += [self._w_logvar, self._b_logvar]
        else:
            self._weight = self._w_mu
            self._bias = self._b_mu
        

        self.reset_parameters()


    def _apply(self, fn):
        return super(Parameters, self)._apply(fn)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self._w_mu.size(1))
        logvar_init = math.log(stdv) * 2
        self._w_mu.data.uniform_(-stdv, stdv)
        self._b_mu.data.uniform_(-stdv, stdv)
        if self._type  == ModelType.BBB or self._type.value in [2,3] :
            self._w_logvar.data.fill_(logvar_init)
            self._b_logvar.data.fill_(logvar_init)
       
    
    
    @property
    def values(self):
        return self._values
    
    @property
    def store_samples(self):
        return self._store_samples

    @store_samples.setter
    def store_samples(self,store):
        if store is False: self.clear_samples()
        self._store_samples = store

    @property
    def sampling(self):
        return self._sampling

    @sampling.setter
    def sampling(self,sampling):
        self._sampling = sampling

    @property
    def weight(self):
        assert self._weight is not None
        if isinstance(self._weight, GaussianDistribution):
            # print(self._sampling, self._store_samples)
            if self._sampling: return self._weight.sample(self._store_samples)
            else: return self._weight.mu
        elif  self._type.value in [2,3]: return self._weight, self._w_logvar
        return self._weight

    @property
    def bias(self):
        assert self._bias is not None
        if isinstance(self._bias, GaussianDistribution):
            if self._sampling: return self._bias.sample(self._store_samples)
            else: return self._bias.mu
        elif  self._type.value in [2,3]: return self._bias, self._b_logvar
        return self._bias
    
    def clear_samples(self):
        if isinstance(self._weight, GaussianDistribution):
            self._weight.clear_samples()
            self._bias.clear_samples()
    
    def get_kl(self):
        kl = 0
        if isinstance(self._weight, GaussianDistribution):
            kl += self._weight.kl_div()
            kl += self._bias.kl_div()
        if kl == 0:return 1e-18
            
        return kl



class BaseLayer(nn.Module):
    def __init__(self, type):
        super(BaseLayer, self).__init__()
        self._type = type
        self._list_params = []


    def register_layer_parameters(self, parameters):
        parameters = parameters if isinstance(parameters,list) else [parameters]
        for i, p in enumerate(parameters):
            for j,value in enumerate(p.values):
                setattr(self,"parameter_{}_{}".format(len(self._list_params)+i,j), value)

        self._list_params.extend(parameters)

    def _apply(self, fn):
        return super(BaseLayer, self)._apply(fn)

    # def _log_variational_posterior(self):
    #     """
    #     theta is from a multivariate gaussian with diagnol covariance
    #     return the loglikelihood.
    #     :param weights: a list of weights
    #     :param means: a list of means
    #     :param logvars: a list of logvars
    #     :return ll: loglikelihood sum over list
    #     """
    #     assert len(self._list_params) > 0
    #     if  len(self._list_params) == 1: return self._list_params[0].log_posterior()
    #     return torch.cat([p.log_posterior() for p in self._list_params])

    # def _log_prior(self):
    #     """
    #     :param weights: a list of weights
    #     :param logstd: a list of means
    #     :param logstd: number
    #     :return ll: likelihood sum over the list
    #     """
    #     assert len(self._list_params) > 0
    #     if  len(self._list_params) == 1: return self._list_params[0].log_prior()
    #     return torch.cat([p.log_prior() for p in self._list_params])
    
    def _bbb_kl(self):
        assert len(self._list_params) > 0
        kl = 0
        for p in self._list_params:
             kl += p.get_kl()
        return kl

    def _variational_dropout_kl(self):
        assert len(self._list_params) > 0
        #It does not use the comple formula. It uses the described in the appendix C  of the Paper)
        #c = [ 1.16145124, 1.50204118, 0.58629921]
        kl = 0.0
        #total = 1e-18
        for p in self._list_params:
            _,logalpha = p.weight
            _,logalpha_b = p.bias
            #alpha_t = F.softplus(logalpha, beta=1, threshold=20) +1e-16
            #alpha_b = F.softplus(logalpha_b, beta=1, threshold=20) + 1e-16

            #logalpha = (0.5*logvar).exp() / (theta**2 + 1e-16)
            #alpha_b = (0.5*logvar_b).exp() / (bias**2 + 1e-16)
#
            
            logalpha = self.clip(logalpha, (-7,-1e-6))
            logalpha_b = self.clip(logalpha_b, (-7,-1e-6))

            kl += (-0.5*logalpha).sum() + (-0.5*logalpha_b).sum()
            
        #print(kl)
            #total = alpha.numel()
        #print(kl)
        return kl if isinstance(kl,torch.Tensor) else 0.0

    @staticmethod
    def clip(input, to=(0.0001,0.9999)):
        return torch.clamp(input, to[0], to[1])


    def clear_samples(self):
        for p in self._list_params: p.clear_samples()

    def get_kl(self):
        kl = 0
        if self._type == ModelType.DET: return kl
        if self._type == ModelType.MC_DROP: return kl
        if self._type == ModelType.BBB: kl = self._bbb_kl()
        else: kl= self._variational_dropout_kl()
        self.clear_samples()
        return kl

    def sampling(self,sampling,store):
        for p in self._list_params: 
            p.sampling = sampling
            p.store_samples = store


    def train(self, mode=True):
        self.sampling(mode,mode)
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self


    def eval(self):
        return self.train(False)
    



class Linear(BaseLayer):
    """
    adapted from torch.nn.Linear
    with a Gaussian as prior
    """
    def __init__(self, in_features, out_features, mu = 0, logstd1 = -1, logstd2 = -2, pi = 0.5, type = ModelType.BBB, dropout = 0.0):
        super(Linear, self).__init__(type = type)
        self._params = Parameters(size_w = (out_features,in_features), \
                                      size_b = (out_features), \
                                      mu = mu, \
                                      logstd1=logstd1, \
                                      logstd2=logstd2, \
                                      pi=pi, \
                                      type=type)

        
        self.dropout = dropout
        self.register_layer_parameters(self._params)

    def _apply(self, fn):
        return super(Linear, self)._apply(fn)
    
    def set_dropout(self,value):
        self.dropout = value
    #Do remember to clean samples.
    def forward(self, inputs):
        if self._type == ModelType.BBB:
            weight = self._params.weight
            bias = self._params.bias
            return F.linear(inputs, weight, bias)
            
        elif self._type == ModelType.DET:
            weight = self._params.weight
            bias = self._params.bias
            layer = F.linear(inputs, weight, bias)
            if (self.dropout in [0.0,1.0]) or not self.training: return layer
            return F.dropout(layer, p=self.dropout, training=True)

        elif self._type == ModelType.MC_DROP:
            weight = self._params.weight
            bias = self._params.bias
            layer = F.linear(inputs, weight, bias)
            if (self.dropout in [0.0,1.0]): return layer
            return F.dropout(layer, p=self.dropout, training=True)


        elif self._type == ModelType.VAR_DROP_B_ADAP:
            # Paper:
            # Kingma, Durk P., Tim Salimans, and Max Welling. 
            #     "Variational dropout and the local reparameterization trick." 
            #     Advances in Neural Information Processing Systems. 2015.
            
            theta, logvar_t = self._params.weight
            bias,  logvar_b = self._params.bias
            
            
            if not self.training:
                return F.linear(inputs,theta,bias)
            eps = 1e-8
            #logalpha = logvar - 2 * torch.log(torch.abs(theta) + eps)

            # calculate std
            size = list(inputs.shape)
            A = inputs.view(-1,size[-1])
            
            # Section 3.1
            alpha_t = self.clip(logvar_t.exp())
            alpha_b = self.clip(logvar_b.exp())

            #Equation 6
            theta_mu = torch.mm(A, theta.t())  
            var = torch.mm(A.pow(2), (alpha_t*(theta.pow(2))).t()) + eps
            theta_std = torch.sqrt(var)

            eps_theta = torch.randn_like(theta_mu)
            eps_bias = torch.randn_like(alpha_b)

            B = theta_mu + theta_std * eps_theta  
            bias = bias + alpha_b * eps_bias
            size[-1] = B.size(-1)
            return B.view(size) + bias

        elif self._type == ModelType.VAR_DROP_A_ADAP:

            # Paper:
            # Kingma, Durk P., Tim Salimans, and Max Welling. 
            #     "Variational dropout and the local reparameterization trick." 
            #     Advances in Neural Information Processing Systems. 2015.
            
            theta, logvar_t = self._params.weight
            bias,  logvar_b = self._params.bias
        
            A = inputs
            
            if not self.training:
                return F.linear(A, theta, bias)

            # Section 3.2 
            alpha_t = self.clip(logvar_t.exp())
            alpha_b = self.clip(logvar_b.exp())

            # ---------------- Equation 11 ---------------------------
            size = list(inputs.shape)

            S = np.random.randn((size))
            S = (1 + S*alpha_t.unsqueeze(0)) #N(1,alpha)
            W = S*theta
            W = torch.from_numpy(block_diag(*S))

            S = np.random.randn((A.size(0),*list(alpha.shape)))
            S = (1 + S*alpha_b.unsqueeze(0)) #N(1,alpha)
            b = S*bias
            b = torch.from_numpy(block_diag(*S))

            B = torch.mm(A.view(1,-1),W)
            B = B.view(A.size(0),-1) + bias
            #--------------------------------------------------------
            return B

class Conv2d(BaseLayer):
    """
    adapted from torch.nn.Conv2d
    with a Gaussian as prior
    """


    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        pass
        

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

    def __init__(self, in_features, out_features, kernel_size, strid=1, pedding = 0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', mu = 0, logstd1 = -1, logstd2 = -2, pi = 0.5, type = ModelType.BBB, dropout = 0.2):
        super(Conv2d, self).__init__(type = type)
        self.kernel_size = (kernel_size,kernel_size)
        self.stride = (stride,stride)
        self.padding = (padding,padding)
        self.dilation = (dilation,dilation)
        self.groups = groups
        self.dropout = dropout

        self._params = Parameters(size_w = (out_features,in_features//groups,*self.kernel_size), \
                                      size_b = (out_features), \
                                      mu = mu, \
                                      logstd1=logstd1, \
                                      logstd2=logstd2, \
                                      pi=pi, \
                                      type=type)
        
        self.register_layer_parameters(self._params)



    #Do remember to clean samples.
    def _conv2d_dropout(self,input,w,b):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            w, b, self.stride,
                            _pair(0), self.dilation, self.groups)

        layer = F.conv2d(input, w, b, self.stride,\
                        self.padding, self.dilation, self.groups)
        
        if self.dropout==0.0 or self.dropout == 1.0: return layer
        return F.dropout2d(layer, p=self.dropout, training=True)

    def forward(self, inputs):
        if self._type == ModelType.BBB:
            weight = self._params.weight
            bias = self._params.bias
            return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

        elif self._type == ModelType.MC_DROP:
            weight = self._params.weight
            bias = self._params.bias
            return self._conv2d_dropout(input,weight,bias)

        
        elif self._type == ModelType.VAR_DROP_B_ADAP:
            # Paper:
            # Kingma, Durk P., Tim Salimans, and Max Welling. 
            #     "Variational dropout and the local reparameterization trick." 
            #     Advances in Neural Information Processing Systems. 2015.
            
            theta, logvar_t = self._params.weight
            bias,  logvar_b = self._params.bias
            
            
            if not self.training:
                return F.linear(inputs,theta,bias)
                return F.conv2d(inputs, theta, bias=bias, stride=self.stride, padding=self.padding, \
                                dilation=self.dilation, groups=self.groups)

            eps = 1e-8
            #logalpha = logvar - 2 * torch.log(torch.abs(theta) + eps)

            # calculate std
            size = list(inputs.shape)
            A = inputs.view(-1,size[-1])

            alpha_t = self.clip(logvar_t.exp())
            alpha_b = self.clip(logvar_b.exp())

            #Equation 6
            theta_mu = F.conv2d(A, theta.t(), bias=None, stride=self.stride, padding=self.padding, \
                                    dilation=self.dilation, groups=self.groups)

            var = F.conv2d(A.pow(2), (alpha_t*(theta.pow(2))).t(), bias=None, stride=self.stride, \
                                    padding=self.padding, dilation=self.dilation, groups=self.groups) + eps
            
            theta_std = torch.sqrt(var)

            eps_theta = torch.randn_like(theta_mu)
            eps_bias = torch.randn_like(alpha_b)

            B = theta_mu + theta_std * eps_theta  
            bias = bias + alpha_b * eps_bias
            size[-1] = B.size(-1)
            return B.view(size) + bias

        elif self._type == ModelType.VAR_DROP_A_ADAP:

            # Paper:
            # Kingma, Durk P., Tim Salimans, and Max Welling. 
            #     "Variational dropout and the local reparameterization trick." 
            #     Advances in Neural Information Processing Systems. 2015.
            
            theta, logvar_t = self._params.weight
            bias,  logvar_b = self._params.bias
        
            A = inputs
            
            if not self.training:
                return F.linear(A, theta, bias)

            # Section 3.1 
            alpha_t = self.clip(logvar_t.exp())
            alpha_b = self.clip(logvar_b.exp())

            # ---------------- Equation 11 ---------------------------
            size = list(inputs.shape)

            S = np.random.randn((size))
            S = (1 + S*alpha_t.unsqueeze(0)) #N(1,alpha)
            W = S*theta
            W = torch.from_numpy(block_diag(*S))

            S = np.random.randn((A.size(0),*list(alpha.shape)))
            S = (1 + S*alpha_b.unsqueeze(0)) #N(1,alpha)
            b = S*bias
            b = torch.from_numpy(block_diag(*S))

            B = F.conv2d(A.view(1,-1),W, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
            
            B = B.view(A.size(0),-1) + bias
            #--------------------------------------------------------
            return B



class LSTM(nn.Module):

    def __init__(self,input_size, hidden_size,
                 num_layers=1, batch_first=False,
                 dropout=0, mu = 0, logstd1 = -1, logstd2 = -2, pi = 0.5, type = ModelType.BBB, peephole = False):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.peephole = peephole
        self.batch_first = batch_first
        self.dropout = dropout
        gate_size = 4 * hidden_size
        self._dense_layers_input = nn.ModuleList()
        self._dense_layers_hidden = nn.ModuleList()
        hidden_size = hidden_size*2 if self.peephole else hidden_size
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self._dense_layers_input.append(Linear(layer_input_size, gate_size,mu = mu, logstd1=logstd1, logstd2=logstd2, pi=pi, dropout = dropout, type = type))
            self._dense_layers_hidden.append(Linear(hidden_size, gate_size, mu = mu,  logstd1=logstd1,logstd2=logstd2,pi=pi, dropout = dropout, type = type))
    
    def set_dropout(self,dropout):
        for li,lh in zip(self._dense_layers_input, self._dense_layers_hidden):
            li.dropout = dropout
            lh.dropout = dropout
            


    def lstm_peephole_cell(self, x, hidden, dense_i, dense_h ):
        h, c = hidden
        # Linear mappings
        peephole = torch.cat((h,c),-1).contiguous()
        print(dense_h.weight.shape, dense_i.weight.shape, x.shape)
        preact = dense_i(x) + dense_h(peephole) 
        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        f = gates[:, self.hidden_size:2 * self.hidden_size]
        i = gates[:, :self.hidden_size]
        g = preact[:, 3 * self.hidden_size:].tanh()
        o = gates[:, -self.hidden_size:]
       
        c_t = c*f + i*g
        h_t = o*c_t.tanh()

        #if do_dropout:
            #h_t = F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            
        return h_t, c_t

    def lstm_cell(self, x, hidden, dense_i, dense_h ):
        #do_dropout = self.training and self.dropout > 0.0
        h, c = hidden
        # Linear mappings
        preact = dense_i(x) + dense_h(h)
        # activations
        gates = preact[:, :3 * self.hidden_size].sigmoid()
        f = gates[:, self.hidden_size:2 * self.hidden_size]
        i = gates[:, :self.hidden_size]
        g = preact[:, 3 * self.hidden_size:].tanh()
        o = gates[:, -self.hidden_size:]
       
        c_t = c*f + i*g
        h_t = o*c_t.tanh()

        #if do_dropout:
            #h_t = F.dropout(h_t, p=self.dropout, training=self.training, inplace=True)
            
        return h_t, c_t

  

    def forward(self, input, hidden=None): 
        
        if not self.batch_first:
            input= input.permute(1, 0, 2)

        if isinstance(input, PackedSequence):
            input, batch_sizes = input
            batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            batch_size = input.size(0)
           
        #initialize hidden state if it is None
        if hidden is None:
            zeros = torch.zeros(self.num_layers,batch_size, self.hidden_size).to(input.device)
            hidden = (zeros,zeros) 
        
        #Iterate over the input sequence and return the last layer's result
        seq_len = input.size(1) 
        cur_layer_input = input
        layers_last_hidden = []
        lstm_cell = self.lstm_peephole_cell if self.peephole else self.lstm_cell
        for layer in range(self.num_layers):
            dense_i, dense_h = self._dense_layers_input[layer],  self._dense_layers_hidden[layer]
            output_seq = []
            hx = (hidden[0][layer], hidden[1][layer])
            #Iterate over sequence
            for t in range(seq_len):
                x = cur_layer_input[:,t,:]
                h, c = lstm_cell(x, hx, dense_i, dense_h)
                hx = (h,c)
                output_seq.append(h)
            layers_last_hidden.append(hx)
            output_layer = torch.stack(output_seq, dim=1)
            cur_layer_input = output_layer
        h = torch.stack([ho[0] for ho in layers_last_hidden],dim=0)
        c = torch.stack([ho[1] for ho in layers_last_hidden],dim=0)
        return output_layer, (h,c) 
        




class GRU(nn.Module):

    def __init__(self,input_size, hidden_size,
                 num_layers=1, batch_first=False,
                 dropout=0, mu = 0, logstd1 = -1, logstd2 = -2, pi = 0.5, type = ModelType.BBB):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        gate_size = 3 * hidden_size
        self._dense_layers_input = nn.ModuleList()
        self._dense_layers_hidden1 = nn.ModuleList()
        self._dense_layers_hidden2 = nn.ModuleList()


        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self._dense_layers_input.append(Linear(layer_input_size,gate_size,mu = mu, logstd1=logstd1, logstd2=logstd2, pi=pi, dropout = dropout, type = type))
            self._dense_layers_hidden1.append(Linear(hidden_size,2*hidden_size, mu = mu,  logstd1=logstd1,logstd2=logstd2,pi=pi, dropout = dropout, type = type))
            self._dense_layers_hidden2.append(Linear(hidden_size,hidden_size, mu = mu,  logstd1=logstd1,logstd2=logstd2,pi=pi, dropout = dropout, type = type))



    def gru_cell(self, x, h, dense_i, dense_h, dense_h2 ):

        preact_i = dense_i(x).chunk(3,1)
        preact_h = dense_h(h).chunk(2,1)

        z = (preact_i[0] + preact_h[0]).sigmoid()
        r = (preact_i[1] + preact_h[1]).sigmoid()
        one = torch.ones_like(z)
        h = z*h + (one-z) * (preact_i[2] + dense_h2(r*h)).tanh()
            
        return h

  

    def forward(self, input, hidden=None): 
        
        if not self.batch_first:
            input= input.permute(1, 0, 2)

        if isinstance(input, PackedSequence):
            input, batch_sizes = input
            batch_size = batch_sizes[0]
        else:
            batch_sizes = None
            batch_size = input.size(0)
           
        #initialize hidden state if it is None
        if hidden is None:
            hidden = torch.zeros(self.num_layers,batch_size, self.hidden_size).to(input.device)
        
        #Iterate over the input sequence and return the last layer's result
        seq_len = input.size(1) 
        cur_layer_input = input
        layers_last_hidden = []
        for layer in range(self.num_layers):
            dense_i, dense_h, dense_h2  = self._dense_layers_input[layer],  self._dense_layers_hidden1[layer], self._dense_layers_hidden2[layer]
            output_seq = []
            h = hidden[layer]
            #Iterate over sequence
            for t in range(seq_len):
                x = cur_layer_input[:,t,:]
                h = self.gru_cell(x, h, dense_i, dense_h, dense_h2)
                output_seq.append(h)
            layers_last_hidden.append(output_seq[-1])
            output_layer = torch.stack(output_seq, dim=1)
            cur_layer_input = output_layer
        h = torch.stack([ho for ho in layers_last_hidden],dim=0)
        return output_layer, h
        


# input_size = 32, output_size = 12, hidden_dim = 64, n_layers = 2, std = -2, epoch= 1000, lr = 0.005, batch = 32, sequence = 32, clip = 0.4, max=128
