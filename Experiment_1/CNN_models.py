import torch
from Experiment_1 import calc_expectation
import tensorly
import tltorch


class uncomp_model(torch.nn.Module):
    def __init__(self,in_channels   :int
                 , out_channels     :int
                 , kernel_size      :int | tuple
                 , stride           :int | tuple        = 1
                 , padding          :int | tuple |str   = 0
                 , dilation         :int | tuple        = 1
                 , groups           :int                = 1
                 , bias             :bool               = True              
                 , padding_mode     :str                = 'zeros'     
                 , device           :str                = None             
                 , dtype            :torch.dtype        = torch.float32):
        super().__init__()

        self._in_channels    = in_channels
        self._out_channels   = out_channels
        self._kernel_size    = kernel_size
        self._stride         = stride
        self._padding        = padding
        self._dilation       = dilation
        self._bias           = bias
        self._padding_mode   = padding_mode
        self._dtype          = dtype


        self.encoder = torch.nn.Conv2d(in_channels=in_channels
                                       , out_channels=out_channels
                                       , kernel_size=kernel_size
                                       , stride=stride
                                       , padding=padding
                                       , dilation=dilation
                                       , bias=bias
                                       , padding_mode=padding_mode
                                       , device = device
                                       , dtype=dtype
                                       , groups=groups)
        
    def MAC_and_RAM(self, image):
        return calc_expectation.MAC_and_ram_estimation_2d(self._in_channels, self._out_channels, self._kernel_size, image, 'uncomp', self._stride, self._padding, self._dilation, bits_per_element=torch.finfo(self._dtype).bits)

    def forward(self,x):
        return self.encoder(x)
    

class cp_tensorly_model(torch.nn.Module):
    def __init__(self,in_channels   :int
                 , out_channels     :int
                 , kernel_size      :int | tuple
                 , rank             :int | float 
                 , stride           :int | tuple        = 1
                 , padding          :int | tuple |str   = 0
                 , dilation         :int | tuple        = 1
                 , groups           :int                = 1
                 , bias             :bool               = True              
                 , padding_mode     :str                = 'zeros'     
                 , device           :str                = None            
                 , dtype            :torch.dtype        = torch.float32
                 , implementation   :str                = 'factorized'):
        super().__init__()

        self._in_channels    = in_channels
        self._out_channels   = out_channels
        self._kernel_size    = kernel_size
        self._rank           = rank
        self._implementation = implementation   
        self._stride         = stride
        self._padding        = padding
        self._dilation       = dilation
        self._bias           = bias
        self._padding_mode   = padding_mode
        self._dtype          = dtype


        self.encoder = tltorch.FactorizedConv.from_conv(torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode,device,dtype)
                                                        ,rank
                                                        ,implementation = implementation
                                                        ,factorization= 'cp'
                                                        ,decompose_weights=True)

    def MAC_and_RAM(self, image):
        if self._implementation == 'factorized':
            method = 'cp'
        elif (self._implementation == 'reconstructed'):
            method = 'uncomp'
        else:
            NotImplementedError(r'For this implementation of the FactorizedConv, no MAC and RAM are worked out')
    
        return calc_expectation.MAC_and_ram_estimation_2d(self._in_channels, self._out_channels, self._kernel_size, image, method, self._stride, self._padding, self._dilation, bits_per_element=torch.finfo(self._dtype).bits,rank=self._rank)
    def forward(self, x):
        return self.encoder(x)
    


    
    