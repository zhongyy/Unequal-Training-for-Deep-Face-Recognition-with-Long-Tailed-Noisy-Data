# THIS FILE IS FOR EXPERIMENTS, USE image_iter.py FOR NORMAL IMAGE LOADING.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import logging
import sys
import numbers
import math
import sklearn
import datetime
import numpy as np
import cv2

import mxnet as mx
from mxnet import ndarray as nd
from mxnet import io
from mxnet import recordio
from mxnet import image
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_preprocess
import multiprocessing

logger = logging.getLogger()

class FaceImageIter(io.DataIter):

    def __init__(self, mx_pretrained = None, ctx = None, ctx_num = 2, path_imgrec = None, mean = None,
                 data_shape = None, batch_size = 90, batch_size_mining = 1, bin_dir = None, threshold = 0.007,
                 shuffle = True, rand_mirror = True, data_name = 'data', label_name = 'softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrec
        if path_imgrec:
            logging.info('loading recordio %s...', path_imgrec)
            path_imgidx = path_imgrec[0:-4]+".idx"
            self.imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')  # pylint: disable=redefined-variable-type
            s = self.imgrec.read_idx(0)
            header, _ = recordio.unpack(s)
            if header.flag>0:
              print('header label', header.label)
              self.header0_1 = (int(header.label[0]), int(header.label[1]))
              self.imgidx = range(1, int(header.label[0]))
              self.id2range = {}
              self.seq_identity = range(int(header.label[0]), int(header.label[1]))
              for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                header, _ = recordio.unpack(s)
                a,b = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (a,b)
                count = b-a
              print('id2range', len(self.id2range))
            else:
              self.imgidx = list(self.imgrec.keys)

            self.seq = []
            self.oseq = self.imgidx
            print("ori samples: ",len(self.oseq))

        self.threshold = threshold
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        self.provide_data_mining = [(data_name, (batch_size_mining,) + data_shape)]
        self.provide_label = [(label_name, (batch_size,))]
        self.provide_label_mining = [(label_name, (batch_size_mining,))]
        self.batch_size_mining = batch_size_mining
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.check_data_shape(data_shape)
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.cur = 0
        self.is_init = False
        self.mx_pretrained = mx_pretrained
        self.mx_model = None
        self.ctx_num = ctx_num
        self.ctx = ctx
        self.model_t = None
        self.oseq_cur = 0
        self.save = 0
        self.bin_dir = bin_dir
        self.first_reset = 1
        self.nbatch = 0
        self.mean = mean
        self.nd_mean = None
        if self.mean:
          self.mean = np.array(self.mean, dtype=np.float32).reshape(1,1,3)
          self.nd_mean = mx.nd.array(self.mean).reshape((1,1,3))

    def check_data_shape(self, data_shape):
        """Checks if the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """Checks if the input data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def postprocess_data(self, datum):
        """Final postprocessing step before image is loaded into the batch."""
        return nd.transpose(datum, axes=(2, 0, 1))

    def imdecode(self, s):
        """Decodes a string or byte string to an NDArray.
        See mx.img.imdecode for more details."""
        img = mx.image.imdecode(s)  # mx.ndarray
        return img

    def time_reset(self):
      self.time_now = datetime.datetime.now()

    def time_elapsed(self):
      time_now = datetime.datetime.now()
      diff = time_now - self.time_now
      return diff.total_seconds()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        if self.first_reset == 1:
            print("first reset")
            #all_layers = self.mx_model.symbol.get_internals()
            # print('all_layers: ',all_layers)
            if self.model_t is None:
                vec = self.mx_pretrained.split(',')
                assert len(vec) > 1
                prefix = vec[0]
                epoch = int(vec[1])
                print('loading', prefix, epoch)
                sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
                all_layers = sym.get_internals()
                print('all_layers:',all_layers)
                sym = all_layers['blockgrad1_output']
                self.model_t = mx.mod.Module(symbol=sym, context=self.ctx)
                self.model_t.bind(data_shapes=self.provide_data_mining, label_shapes=self.provide_label_mining)
                self.model_t.set_params(arg_params, aux_params)
            ba = 0
            tag = []
            data = nd.zeros(self.provide_data_mining[0][1])
            label = nd.zeros(self.provide_label_mining[0][1])
            outfilew = os.path.join(self.bin_dir, "%d_noiselist.txt" % (self.save))
            with open(outfilew, 'w') as fp:
                while True:
                    bb = min(ba + self.batch_size_mining, len(self.oseq))
                    print("start bb,ba",ba,bb)
                    if ba >= bb:
                        break
                    for i in xrange(ba, bb):
                        _idx = self.oseq[i]
                        s = self.imgrec.read_idx(_idx)
                        header, img = recordio.unpack(s)
                        img = self.imdecode(img)
                        data[i - ba][:] = self.postprocess_data(img)

                        label0 = header.label
                        if not isinstance(label0, numbers.Number):
                            label0 = label0[0]
                        # print('label0', label0)
                        label[i - ba][:] = label0
                        tag.append((int(label0), _idx))

                    db = mx.io.DataBatch(data=(data,), label=(label,))
                    self.model_t.forward(db, is_train=False)
                    net_out = self.model_t.get_outputs()
                    net_P = mx.nd.softmax(net_out[0], axis=1)
                    net_P = net_P.asnumpy()
                    for ii in range(bb-ba):
                        #print('label:',label[ii])
                        #print('tag:',tag[ii][0])
                        P=net_P[ii]
                        #print(P)
                        #print(max(P))
                        if max(P)<self.threshold:
                            line = '%d %d %s %s\n' % (tag[ii][0], tag[ii][1], max(P), P[tag[ii][0]])
                            fp.write(line)
                        else:
                            self.seq.append(tag[ii][1])
                    tag=[]
                    ba = bb
            self.save += 1
            print("Initialize done: ",len(self.oseq),len(self.seq),len(self.oseq)-len(self.seq))
            self.first_reset += 1
        else:
            print('call reset()')
            self.cur = 0
            if self.shuffle:
              random.shuffle(self.seq)
            self.first_reset += 1


    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq is not None:
          while True:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
              s = self.imgrec.read_idx(idx)
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
              return label, img, None, None
            else:
              label, fname, bbox, landmark = self.imglist[idx]
              return label, self.read_image(fname), bbox, landmark
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None

    def next(self):
        if not self.is_init:
          self.reset()
          self.is_init = True
        """Returns the next batch of data."""
        #print('in next', self.cur, self.labelcur)
        self.nbatch+=1
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size, c, h, w))
        if self.provide_label is not None:
          batch_label = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size:
                label, s, bbox, landmark = self.next_sample()
                _data = self.imdecode(s)
                _data = _data.astype('float32')
                _data = image.RandomGrayAug(.2)(_data)
                if random.random() < 0.2:
                    _data = image.ColorJitterAug(0.2, 0.2, 0.2)(_data)
                if self.rand_mirror:
                  _rd = random.randint(0,1)
                  if _rd==1:
                    _data = mx.ndarray.flip(data=_data, axis=1)
                if self.nd_mean is not None:
                    _data = _data.astype('float32')
                    _data -= self.nd_mean
                    _data *= 0.0078125
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                #print('aa',data[0].shape)
                #data = self.augmentation_transform(data)
                #print('bb',data[0].shape)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    #print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i<batch_size:
                raise StopIteration

        return io.DataBatch([batch_data], [batch_label], batch_size - i)
