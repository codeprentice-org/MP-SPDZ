#!/usr/bin/env python3

import sys
from functools import reduce
import operator
import math
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import numpy
import ml

first = True
paddings = {}

def output(layers, named, op, layer, prev_input=True):
    global first
    named[op.name] = layer
    layers.append(named[op.name])
    if prev_input and not first:
        named[op.name].inputs = [named[op.inputs[0].name[:-2]]]
    first = False

def link(dest, source, named):
    named[dest.name] = named[source.name]

def source(dest, named):
    named[dest.name] = None

def activate_bias(op, named):
    named[op.name].input_bias = True

def get_shape(shape):
    res = []
    for x in shape:
        try:
            res.append(int(x))
        except:
            res.append(1)
    return res

def get_valid_padding(input_shape, window, strides):
    return [int(math.ceil((x - y + 1) / z))
            for x, y, z in zip(input_shape, window, strides)]

def get_layers_and_named_data(filename):
    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(open(filename, mode='rb').read())
    tf.import_graph_def(graph_def)
    graph = tf.compat.v1.get_default_graph()

    named = {}
    layers = []
    for op in graph.get_operations():
        if op.inputs:
            shape = get_shape(op.inputs[0].shape)
        else:
            shape = None
        t = op.type
        if t in ('VariableV2', 'Const'):
            pass
        elif t in ('Reshape', 'Squeeze'):
            link(op, op.inputs[0].op, named)
        elif t == 'Placeholder':
            source(op, named)
        elif t == 'MatMul':
            assert reduce(operator.mul, shape) == op.inputs[1].shape[0]
            output(layers, named, op, ml.Dense(1, op.inputs[1].shape[0], op.inputs[1].shape[1]))
            shape = [1, int(op.inputs[1].shape[1])]
        elif t == 'Conv2D':
            strides = op.get_attr('strides')
            assert len(strides) == 4
            assert strides[0] == 1
            assert strides[3] == 1
            strides = tuple(strides[1:3])
            input_shape = get_shape(op.inputs[0].shape)
            assert len(input_shape) == 4
            window = [int(x) for x in op.inputs[1].shape]
            padding = op.get_attr('padding').decode('u8')
            if padding not in ('SAME', 'VALID'):
                padding = get_shape(padding)
            if op.inputs[0].op.name in paddings:
                assert padding == 'VALID'
                input_shape = get_shape(op.inputs[0].op.inputs[0].shape)
                p = paddings.pop(op.inputs[0].op.name)
                for i in 0, 6:
                    assert p[i] == 0
                padding = [p[2], p[4]]
            output_shape = get_shape(op.outputs[0].shape)
            assert len(output_shape) == 4
            output(layers, named, op, ml.FixConv2d(input_shape, tuple(window), (window[3],), output_shape, strides, padding, True,
                inputs=[named[op.inputs[0].op.name]]))
        elif t == 'Add' and op.inputs[1].op.type != 'VariableV2':
            output(layers, named, op, ml.Add([named[x.op.name] for x in op.inputs]), False)
        elif t in ('Add', 'BiasAdd'):
            assert op.inputs[0].op.type in ('MatMul', 'Conv2D')
            activate_bias(op.inputs[0].op, named)
            link(op, op.inputs[0].op, named)
        elif t == 'Relu':
            assert len(op.inputs) == 1
            output(layers, named, op, ml.Relu(shape, inputs=[named[op.inputs[0].op.name]]))
        elif t == 'Square':
            output(layers, named, op, ml.Square(shape))
        elif t == 'MaxPool':
            strides = op.get_attr('strides')
            ksize = op.get_attr('ksize')
            padding = str(op.get_attr('padding').decode('u8'))
            output(layers, named, op, ml.MaxPool(shape, strides, ksize, padding))
        elif t == 'AvgPool':
            filter_size = op.get_attr('ksize')
            assert len(filter_size) == 4
            assert filter_size[0] == 1
            assert filter_size[-1] == 1
            input_shape = get_shape(op.inputs[0].shape)
            strides = get_shape(op.get_attr('strides'))
            assert strides[0] == 1
            assert strides[3] == 1
            padding = op.get_attr('padding').decode('u8')
            if padding == 'VALID':
                output_shape = get_valid_padding(input_shape, filter_size, strides)
            elif padding == 'SAME':
                output_shape = [int(math.ceil(x / y))
                                for x, y in zip(input_shape, filter_size)]
            else:
                raise Exception('unknown padding type: %s' % padding)
            output(layers, named, op, ml.FixAveragePool2d(input_shape, output_shape, filter_size[1:3], strides[1:3]))
        elif t == 'ArgMax':
            assert len(op.inputs) == 2
            shape = get_shape(op.inputs[0].shape)
            dim = int(op.inputs[1].op.get_attr('value').int_val[0])
            for i in range(1, len(shape)):
                if i != dim:
                    assert shape[i] == 1
            output(layers, named, op, ml.Argmax((1, shape[dim])))
        elif t == 'ConcatV2':
            assert len(op.inputs) == 3
            dim = int(op.inputs[2].op.get_attr('value').int_val[0])
            output(layers, named, op, ml.Concat([named[x.name[:-2]] for x in op.inputs[:2]], dim), prev_input=False)
        elif t == 'FusedBatchNorm':
            output(layers, named, op, ml.FusedBatchNorm(get_shape(op.inputs[0].shape), inputs=[named[op.inputs[0].op.name]]))
        elif t == 'Pad':
            paddings[op.name] = numpy.fromstring(op.inputs[1].op.get_attr('value').
                                                tensor_content, 'int32').tolist()
            link(op, op.inputs[0].op, named)
        elif t == 'FusedBatchNormV3':
            # This treats the input the same as FusedBatchNorm (might be erroneous)
            # print("Inputs: ", named[op.inputs[0].op.name])
            # print("Shape: ", get_shape(op.inputs[0].shape))
            # _layer =  ml.FusedBatchNorm(
            #             get_shape(op.inputs[0].shape), 
            #             inputs=[named[op.inputs[0].op.name]])
            # output(layers, named, op, _layer)
            raise Exception('unknown type: %s' % t)
        else:
            raise Exception('unknown type: %s' % t)

    if paddings:
        raise Exception('padding layers only supported before valid convolution:',
                        paddings)
    return (layers, named)
