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

    def __init__(self, mx_model = None, ctx = None, ctx_num = 2,
                 data_shape = None, batch_size1 = 90, path_imgrec1 = None, batchsize_id = 90,
                 batch_size2 = 90, path_imgrec2 = None, images_per_identity = 3, interclass_bag_size =3600,
                 shuffle = True, rand_mirror = True,
                 data_name = 'data', label_name = 'softmax_label', **kwargs):
        super(FaceImageIter, self).__init__()
        assert path_imgrec1
        if path_imgrec1:
            logging.info('loading recordio %s...', path_imgrec1)
            path_imgidx1 = path_imgrec1[0:-4]+".idx"
            self.imgrec1 = recordio.MXIndexedRecordIO(path_imgidx1, path_imgrec1, 'r')  # pylint: disable=redefined-variable-type
            s1 = self.imgrec1.read_idx(0)
            header1, _ = recordio.unpack(s1)
            if header1.flag>0:
              print('header0_1 label', header1.label)
              self.header0_1 = (int(header1.label[0]), int(header1.label[1]))
              #assert(header.flag==1)
              self.imgidx1 = range(1, int(header1.label[0]))
              self.id2range1 = {}
              self.seq_identity1 = range(int(header1.label[0]), int(header1.label[1]))
              for identity1 in self.seq_identity1:
                s1 = self.imgrec1.read_idx(identity1)
                header1, _ = recordio.unpack(s1)
                a1,b1 = int(header1.label[0]), int(header1.label[1])
                self.id2range1[identity1] = (a1,b1)
                count1 = b1-a1
                #for ii in xrange(a1,b1):
                #  self.idx2flag1[ii] = count1
              print('id2range1', len(self.id2range1))
            else:
              self.imgidx1 = list(self.imgrec1.keys)
            if shuffle:
              self.seq1 = self.imgidx1
              self.oseq1 = self.imgidx1
              print(len(self.seq1))
            else:
              self.seq1 = None

        assert path_imgrec2
        if path_imgrec2:
            logging.info('loading recordio %s...', path_imgrec2)
            path_imgidx2 = path_imgrec2[0:-4] + ".idx"
            self.imgrec2 = recordio.MXIndexedRecordIO(path_imgidx2, path_imgrec2,
                                                      'r')  # pylint: disable=redefined-variable-type
            s2 = self.imgrec2.read_idx(0)
            print(self.imgrec2)
            header2, _ = recordio.unpack(s2)
            if header2.flag > 0:
                print('header0_2 label', header2.label)
                self.header0_2 = (int(header2.label[0]), int(header2.label[1]))
                # assert(header.flag==1)
                self.imgidx2 = range(1, int(header2.label[0]))
                self.id2range2 = {}
                self.seq_identity2 = range(int(header2.label[0]), int(header2.label[1]))
                for identity2 in self.seq_identity2:
                    s2 = self.imgrec2.read_idx(identity2)
                    header2, _ = recordio.unpack(s2)
                    a2, b2 = int(header2.label[0]), int(header2.label[1])
                    self.id2range2[identity2] = (a2, b2)
                    count2 = b2 - a2
                    # for ii in xrange(a1,b1):
                    #  self.idx2flag1[ii] = count1
                print('id2range2', len(self.id2range2))
            else:
                self.imgidx2 = list(self.imgrec2.keys)
            self.seq2 = None
            self.oseq2 = None
            #print(len(self.seq2))


        self.images_per_identity = images_per_identity
        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size1 + batchsize_id,) + data_shape)]
        self.provide_data2 = [(data_name, (batch_size2,) + data_shape)]
        self.batch_size1 = batch_size1
        self.batchsize_id = batchsize_id
        self.interclass_bag_size = interclass_bag_size
        self.batch_size2 = batch_size2
        self.data_shape = data_shape
        self.seq_min_size = batchsize_id*100
        self.shuffle = shuffle
        self.image_size = '%d,%d'%(data_shape[1],data_shape[2])
        self.rand_mirror = rand_mirror
        print('rand_mirror', rand_mirror)
        self.provide_label = [(label_name, (batch_size1 + batchsize_id,))]
        #self.provide_label1 = [(label_name, (batch_size1,))]
        self.provide_label2 = [(label_name, (batch_size2,))]
        #self.provide_label_inter = [(label_name, (2*images_per_identity,))]
        self.cur1 = 0
        self.cur2 = 0
        self.interclass_oseq_cur = 0
        self.nbatch = 0
        self.is_init = False
        self.mx_model = mx_model
        self.ctx_num = ctx_num
        self.ctx = ctx
        self.times = [0.0, 0.0, 0.0]
        self.model_t = None

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

    def interclass_oseq_reset(self):
      #reset self.oseq2 by identities seq
      print('interclass_oseq_reset')
      self.interclass_oseq_cur = 0
      ids = []
      for k in self.id2range2:
        ids.append(k)
      random.shuffle(ids)
      self.oseq2 = []
      for _id in ids:
        v = self.id2range2[_id]
        _list = range(*v)
        random.shuffle(_list)
        if len(_list)>self.images_per_identity:
          _list = _list[0:self.images_per_identity]
        self.oseq2 += _list
      #print('oseq2', self.oseq2)

    def pairwise_dists(self, embeddings):
      nd_embedding_list = []
      for i in xrange(self.ctx_num):
        nd_embedding = mx.nd.array(embeddings, mx.gpu(i))
        nd_embedding_list.append(nd_embedding)
      nd_pdists = []
      pdists = []
      for idx in xrange(embeddings.shape[0]):
        emb_idx = idx%self.ctx_num
        nd_embedding = nd_embedding_list[emb_idx]
        a_embedding = nd_embedding[idx]
        body = mx.nd.broadcast_sub(a_embedding, nd_embedding)
        body = body*body
        body = mx.nd.sum_axis(body, axis=1)
        #print("body: ", body)
        nd_pdists.append(body)
        if len(nd_pdists)==self.ctx_num or idx==embeddings.shape[0]-1:
          for x in nd_pdists:
            pdists.append(x.asnumpy())
          nd_pdists = []
      return pdists

    def pick_interclass(self, embeddings, nrof_images_per_class, batchsize_id):
        #centers
        #print("nrof_images_per_class.len",len(nrof_images_per_class))
        #print("nrof_images_per_class", nrof_images_per_class)
        #print("embeddings.shape", embeddings.shape)
        centers=np.zeros((len(nrof_images_per_class),embeddings.shape[1]))
        for i in xrange(len(nrof_images_per_class)):
            #print('i: ',i)
            if self.images_per_identity==1:
                centers[i] = np.mean([embeddings[i * self.images_per_identity]], axis=0)
            else:
                centers[i] = np.mean([embeddings[i*self.images_per_identity], embeddings[i*self.images_per_identity+1],embeddings[i*self.images_per_identity+2]], axis=0)
        #dist of centers
        pdists = self.pairwise_dists(centers)
        pdists = np.array(pdists)
        pdists = pdists + np.eye(pdists.shape[0])
        #print("pdists",pdists)
        id_sort= np.unravel_index(np.argsort(pdists, axis=None), pdists.shape)
        #id_sort = (np.array([1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2]), np.array([3, 2, 1, 0, 0, 1, 2, 3, 1, 2, 0, 3]))
        id_sel=[]
        id0=0
        while len(id_sel)< batchsize_id//3:
            if id_sort[0][id0] not in id_sel:
                id_sel.append(id_sort[0][id0])
            if id_sort[1][id0] not in id_sel:
                id_sel.append(id_sort[1][id0])
            id0 +=1
        if len(id_sel)>batchsize_id//3:
            id_sel = id_sel[:-1]
        return id_sel
        # interclass
        #interclass = np.zeros(2*self.images_per_identity,embeddings.shape[1])
        #for j in xrange(2):
        #    interclass[j*self.images_per_identity:j*self.images_per_identity+3]=embeddings[id_sel[j]*self.images_per_identity:id_sel[j]*self.images_per_identity+3]
        #return interclass

    def interclass_reset(self):
        self.seq2 = []
        self.oseq2 = []
        while len(self.seq2) < self.seq_min_size:
            self.time_reset()
            embeddings = None
            bag_size = self.interclass_bag_size  # 3600
            batch_size2 = self.batch_size2  # 200
            # data = np.zeros( (bag_size,)+self.data_shape )
            # label = np.zeros( (bag_size,) )
            tag = []
            # idx = np.zeros( (bag_size,) )
            #print('eval %d images..' % bag_size, self.interclass_oseq_cur)  # 3600 0 first time
            #print('interclass time stat', self.times)
            if self.interclass_oseq_cur + bag_size > len(self.oseq2):
                self.interclass_oseq_reset()
                print('eval %d images..' % bag_size, self.interclass_oseq_cur)
            self.times[0] += self.time_elapsed()
            self.time_reset()
            # print(data.shape)
            data = nd.zeros(self.provide_data2[0][1])
            label = nd.zeros(self.provide_label2[0][1])
            ba = 0

            all_layers = self.mx_model.symbol.get_internals()            
            if self.model_t is None:
                symbol_t = all_layers['blockgrad0_output']
                self.model_t = mx.mod.Module(symbol=symbol_t, context=self.ctx, label_names=None)
                self.model_t.bind(data_shapes=self.provide_data2)
                arg_t, aux_t = self.mx_model.get_params()
                self.model_t.set_params(arg_t, aux_t)
            else:
                arg_t, aux_t = self.mx_model.get_params()
                self.model_t.set_params(arg_t, aux_t)

            while True:
                bb = min(ba + batch_size2, bag_size)
                if ba >= bb:
                    break
                # _batch = self.data_iter.next()
                # _data = _batch.data[0].asnumpy()
                # print(_data.shape)
                # _label = _batch.label[0].asnumpy()
                # data[ba:bb,:,:,:] = _data
                # label[ba:bb] = _label
                for i in xrange(ba, bb):
                    _idx = self.oseq2[i + self.interclass_oseq_cur]
                    s = self.imgrec2.read_idx(_idx)
                    header, img = recordio.unpack(s)
                    img = self.imdecode(img)
                    data[i - ba][:] = self.postprocess_data(img)
                    #label[i-ba][:] = header.label
                    #print('header.label', header.label)
                    #print('header.label', header.label.shape)
                    #tag.append((int(header.label), _idx))
                    #print('header.label',header.label)
                    label0 = header.label
                    if not isinstance(label0, numbers.Number):
                        label0 = label0[0]
                    #print('label0', label0)
                    label[i - ba][:] = label0
                    tag.append((int(label0), _idx))
                    # idx[i] = _idx
                #print('tag:' ,tag)
                #print(data,label)

                #db = mx.io.DataBatch(data=(data,), label=(label,))
                #self.mx_model.forward(db, is_train=False)
                #net_out = self.mx_model.get_outputs()

                #print("self.mx_model",self.mx_model)



                db = mx.io.DataBatch(data=(data,), label=(label,))
                self.model_t.forward(db, is_train=False)
                net_out = self.model_t.get_outputs()

                #print('eval for selecting interclasses',ba,bb)
                #print(net_out)
                #print(len(net_out))
                #print(net_out[0].asnumpy())
                net_out = net_out[0].asnumpy()
                #print(len(net_out))
                #print('net_out', net_out.shape)
                if embeddings is None:
                    embeddings = np.zeros((bag_size, net_out.shape[1]))
                #print ("net_out.shape: ", net_out.shape)
                #print("ba,bb: ", ba,bb)
                embeddings[ba:bb, :] = net_out
                ba = bb
            assert len(tag) == bag_size
            self.interclass_oseq_cur += bag_size
            #print("embeddings: ",embeddings)
            embeddings = sklearn.preprocessing.normalize(embeddings)
            self.times[1] += self.time_elapsed()
            self.time_reset()
            nrof_images_per_class = [1]
            for i in xrange(1, bag_size):
                if tag[i][0] == tag[i - 1][0]:
                    nrof_images_per_class[-1] += 1
                else:
                    nrof_images_per_class.append(1)

            id_sel = self.pick_interclass(embeddings, nrof_images_per_class, self.batchsize_id)  # shape=(T,3)
            #print('found interclass', id_sel) #2
            if self.images_per_identity==1:
                for j in xrange(self.batchsize_id // 3):
                    idsel_0 = tag[id_sel[j] * self.images_per_identity][1]
                    self.seq2.append(idsel_0)
            else:
                for j in xrange(self.batchsize_id//3):
                    idsel_0 = tag[id_sel[j]*self.images_per_identity][1]
                    self.seq2.append(idsel_0)
                    idsel_0 = tag[id_sel[j] * self.images_per_identity+1][1]
                    self.seq2.append(idsel_0)
                    idsel_0 = tag[id_sel[j] * self.images_per_identity+2][1]
                    self.seq2.append(idsel_0)
            self.times[2] += self.time_elapsed()

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        if self.seq2 is None:
            print('call reset1()')
            self.cur1 = 0
            if self.shuffle:
                random.shuffle(self.seq1)
            if self.seq1 is None and self.imgrec1 is not None:
                self.imgrec1.reset()
        if self.cur1+self.batch_size1<len(self.seq1):
            print("cur1:", self.cur1)
            print("seq1:", len(self.seq1))
        else:
            print('call reset1()')
            self.cur1 = 0
            if self.shuffle:
              random.shuffle(self.seq1)
            if self.seq1 is None and self.imgrec1 is not None:
                self.imgrec1.reset()
        print('call reset2()')
        self.cur2 = 0
        if self.images_per_identity>0:
            self.interclass_reset()
        if self.seq2 is None and self.imgrec2 is not None:
            self.imgrec2.reset()

    def next_sample(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq1 is not None:
          while True:
            if self.cur1 >= len(self.seq1):
                raise StopIteration
            idx = self.seq1[self.cur1]
            self.cur1 += 1
            if self.imgrec1 is not None:
              s = self.imgrec1.read_idx(idx)
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
              return label, img, None, None
        else: #no
            s = self.imgrec1.read()
            if s is None:
                raise StopIteration
            header, img = recordio.unpack(s)
            return header.label, img, None, None

    def next_sample2(self):
        """Helper function for reading in next sample."""
        #set total batch size, for example, 1800, and maximum size for each people, for example 45
        if self.seq2 is not None:
          #print("self.seq2: ",self.seq2)
          while True:
            if self.cur2 >= len(self.seq2):
                raise StopIteration
            idx = self.seq2[self.cur2]
            #print("cur2: ", self.seq2,idx)
            self.cur2 += 1
            if self.imgrec2 is not None:
              s = self.imgrec2.read_idx(idx)
              header, img = recordio.unpack(s)
              label = header.label
              if not isinstance(label, numbers.Number):
                label = label[0]
              return label, img, None, None

    def next(self):
        if not self.is_init:
            self.reset()
            self.is_init = True
        self.nbatch += 1
        batch_size1 = self.batch_size1
        interclass_size = self.batchsize_id
        c, h, w = self.data_shape
        batch_data = nd.empty((batch_size1+interclass_size, c, h, w))
        batch_data_t = nd.empty((batch_size1 + interclass_size, c, h, w))
        if self.provide_label is not None:
          batch_label = nd.empty(self.provide_label[0][1])
          batch_label_t = nd.empty(self.provide_label[0][1])
        i = 0
        try:
            while i < batch_size1:
                label, s, bbox, landmark = self.next_sample()
                _data = self.imdecode(s)
                _data = _data.astype('float32')
                _data = image.RandomGrayAug(.2)(_data)
                if random.random() < 0.2:
                    _data = image.ColorJitterAug(0.2, 0.2, 0.2)(_data)
                if self.rand_mirror:
                    _rd = random.randint(0, 1)
                    if _rd == 1:
                        _data = mx.ndarray.flip(data=_data, axis=1)
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                for datum in data:
                    assert i < batch_size1, 'Batch size must be multiples of augmenter output length'
                    # print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i < batch_size1:
                raise StopIteration
        try:
            while i < interclass_size+batch_size1:
                label, s, bbox, landmark = self.next_sample2()
                _data = self.imdecode(s)
                _data = _data.astype('float32')
                _data = image.RandomGrayAug(.2)(_data)
                if random.random() < 0.2:
                    _data = image.ColorJitterAug(0.2, 0.2, 0.2)(_data)
                if self.rand_mirror:
                    _rd = random.randint(0, 1)
                    if _rd == 1:
                        _data = mx.ndarray.flip(data=_data, axis=1)
                data = [_data]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                for datum in data:
                    assert i < interclass_size+batch_size1, 'Batch size must be multiples of augmenter output length'
                    # print(datum.shape)
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if i < interclass_size+batch_size1:
                raise StopIteration

        margin = batch_size1//self.ctx_num
        for i in xrange(self.ctx_num):
            batch_data_t[2 * i * margin:(2 * i + 1) * margin][:] = batch_data[i * margin:(i + 1) * margin][:]
            batch_data_t[(2 * i + 1) * margin:2 * (i + 1) * margin][:] = batch_data[batch_size1 + i * margin:batch_size1 + (i + 1) * margin][:]
        for i in xrange(self.ctx_num):
            batch_label_t[2 * i * margin:(2 * i + 1) * margin][:] = batch_label[i * margin:(i + 1) * margin][:]
            batch_label_t[(2 * i + 1) * margin:2 * (i + 1) * margin][:] = batch_label[batch_size1 + i * margin:batch_size1 + (i + 1) * margin][:]
        return io.DataBatch([batch_data_t], [batch_label_t])
