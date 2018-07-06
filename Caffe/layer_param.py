from __future__ import absolute_import
from . import caffe_pb2 as pb
import numpy as np

def _unpair(item,raise_error=True):
    if hasattr(item,'__iter__'):
        for i in item:
            if i!=item[0] and raise_error:
                raise ValueError("number in item {} must be the same".format(item))
        return item[0]
    return item

class Layer_param():
    def __init__(self,name='',type='',top=(),bottom=()):
        self.param=pb.LayerParameter()
        self.name=self.param.name=name
        self.type=self.param.type=type

        self.top=self.param.top
        self.top.extend(top)
        self.bottom=self.param.bottom
        self.bottom.extend(bottom)

    def fc_param(self, num_output, weight_filler='xavier', bias_filler='constant'):
        if self.type != 'InnerProduct':
            raise TypeError('the layer type must be InnerProduct if you want set fc param')
        fc_param = pb.InnerProductParameter()
        fc_param.num_output = num_output
        fc_param.weight_filler.type = weight_filler
        fc_param.bias_filler.type = bias_filler
        self.param.inner_product_param.CopyFrom(fc_param)

    def conv_param(self, num_output, kernel_size, stride=(1), pad=(0,),
                   weight_filler_type='xavier', bias_filler_type='constant',
                   bias_term=True, dilation=None):
        """
        add a conv_param layer if you spec the layer type "Convolution"
        Args:
            num_output: a int
            kernel_size: int list
            stride: a int list
            weight_filler_type: the weight filer type
            bias_filler_type: the bias filler type
        Returns:
        """
        if self.type!='Convolution':
            raise TypeError('the layer type must be Convolution if you want set conv param')
        conv_param=pb.ConvolutionParameter()
        conv_param.num_output=num_output
        conv_param.kernel_size.extend(kernel_size)
        conv_param.stride.extend(stride)
        conv_param.pad.extend(pad)
        conv_param.bias_term=bias_term
        conv_param.weight_filler.type=weight_filler_type
        if bias_term:
            conv_param.bias_filler.type = bias_filler_type
        if dilation:
            conv_param.dilation.extend(dilation)
        self.param.convolution_param.CopyFrom(conv_param)

    def pool_param(self,type='MAX',kernel_size=2,stride=2,pad=None):
        pool_param=pb.PoolingParameter()
        pool_param.pool=pool_param.PoolMethod.Value(type)
        pool_param.kernel_size=_unpair(kernel_size)
        pool_param.stride=_unpair(stride)
        if pad:
            pool_param.pad=pad
        self.param.pooling_param.CopyFrom(pool_param)

    def batch_norm_param(self,use_global_stats=0,moving_average_fraction=None,eps=None):
        bn_param=pb.BatchNormParameter()
        bn_param.use_global_stats=use_global_stats
        if moving_average_fraction:
            bn_param.moving_average_fraction=moving_average_fraction
        if eps:
            bn_param.eps = eps
        self.param.batch_norm_param.CopyFrom(bn_param)

    def add_data(self,*args):
        """Args are data numpy array
        """
        del self.param.blobs[:]
        for data in args:
            new_blob = self.param.blobs.add()
            for dim in data.shape:
                new_blob.shape.dim.append(dim)
            new_blob.data.extend(data.flatten().astype(float))

    def set_params_by_dict(self,dic):
        pass

    def copy_from(self,layer_param):
        pass

def set_enum(param,key,value):
    setattr(param,key,param.Value(value))
