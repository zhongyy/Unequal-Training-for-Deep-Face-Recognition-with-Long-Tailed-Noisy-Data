from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import logging
import pickle
import numpy as np
from data_longtail_gs import FaceImageIter
import mxnet as mx
from mxnet import ndarray as nd
import argparse
import mxnet.optimizer as optimizer
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import verification
import sklearn
#sys.path.append(os.path.join(os.path.dirname(__file__), 'losses'))
#import center_loss


logger = logging.getLogger()
logger.setLevel(logging.INFO)


args = None

class AccMetric(mx.metric.EvalMetric):
  def __init__(self):
    self.axis = 1
    super(AccMetric, self).__init__(
        'acc', axis=self.axis,
        output_names=None, label_names=None)
    self.losses = []
    self.count = 0

  def update(self, labels, preds):
    #print('1: ', preds, labels)
    self.count+=1
    #print('1: ', preds)
    labels = [preds[1]]
    #fc = [preds[2]]
    preds = [preds[4]]  # use softmax output
    #print("fc,preds",fc,preds)
    for label, pred_label in zip(labels, preds):
        #print("label",label)
        #print("pred1", pred_label)
        if pred_label.shape != label.shape:
            pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
        pred_label = pred_label.asnumpy().astype('int32').flatten()
        #print("pred2", pred_label)
        label = label.asnumpy()
        #print("label1", label)
        if label.ndim == 2:
            label = label[:, 0]
        #print("label2", label)
        label = label.astype('int32').flatten()
        #print("label3", label)
        assert label.shape == pred_label.shape
        #print('flat',pred_label.flat)
        self.sum_metric += (pred_label.flat == label.flat).sum()
        self.num_inst += len(pred_label.flat)



class LossValue(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValue, self).__init__(
            name='softmaxloss', axis=self.axis,
            output_names=None, label_names=None)
        self.eps = 0
    def update(self, labels, preds):
        #print("preds: ", preds)
        #loss = preds[4].asnumpy()[0]
        #self.sum_metric += loss
        #self.num_inst += 1.0
        labels = [preds[1]]
        preds = [preds[4]]  # use softmax output
        for label, pred in zip(labels, preds):
            label = label.asnumpy()
            pred = pred.asnumpy()

            #pred = np.transpose(pred)
            #pred = np.exp(pred)/np.sum(np.exp(pred),axis=0)
            #pred = np.transpose(pred)
            #print("pred",pred)
            #pred= np.exp(pred)
            #print("pred", pred.shape)
            label = label.ravel()
            assert label.shape[0] == pred.shape[0]

            prob = pred[np.arange(label.shape[0]), np.int64(label)]
            # print("prob", prob)
            self.sum_metric += -prob.sum()
            self.num_inst += label.shape[0]


class LossValue2(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValue2, self).__init__(
            name='interloss', axis=self.axis,
            output_names=None, label_names=None)
    def update(self, labels, preds):
        #print("cosine_matrix: ", preds[4],preds[-1])
        loss = preds[6].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1


def parse_args():
  parser = argparse.ArgumentParser(description='Train face network')
  # general
  parser.add_argument('--data-dir', default='', help='training set directory')
  parser.add_argument('--data-dir-interclass',default='',help='training interclass set directory')
  parser.add_argument('--prefix', default='../model/model', help='directory to save model.')
  parser.add_argument('--pretrained', default='', help='pretrained model to load')
  parser.add_argument('--ckpt', type=int, default=1, help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
  parser.add_argument('--loss-type', type=int, default=4, help='loss type')
  parser.add_argument('--noise', type=int, default=1, help='loss type')
  parser.add_argument('--verbose', type=int, default=50, help='do verification testing and model saving every verbose batches')
  parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
  parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
  parser.add_argument('--network', default='r50', help='specify network')
  parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
  parser.add_argument('--version-input', type=int, default=1, help='network input config')
  parser.add_argument('--version-output', type=str, default='E', help='network embedding output config')
  parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
  parser.add_argument('--version-act', type=str, default='prelu', help='network activation config')
  parser.add_argument('--use-deformable', type=int, default=0, help='use deformable cnn in network')
  parser.add_argument('--lr', type=float, default=0.001, help='start learning rate')
  parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
  parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
  parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
  parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
  parser.add_argument('--mom', type=float, default=0.9, help='momentum')
  parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
  parser.add_argument('--batch-size1', type=int, default=90, help='batch size in each context')
  parser.add_argument('--batchsize_id', type=int, default=90, help='batch size id for inter')
  parser.add_argument('--bag-size', type=int, default=3600, help='batch size id for inter')
  parser.add_argument('--images-per-identity', type=int, default=3, help='images-per-identity for inter')
  parser.add_argument('--batch-size2',type=int, default=180, help='batch size2')
  parser.add_argument('--margin-inter', type=float, default=1., help='margin for inter')
  parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
  parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
  parser.add_argument('--margin-a', type=float, default=1.0, help='')
  parser.add_argument('--margin-b', type=float, default=0.0, help='')
  parser.add_argument('--easy-margin', type=int, default=0, help='')
  parser.add_argument('--margin', type=int, default=4, help='margin for sphere')
  parser.add_argument('--beta', type=float, default=1000., help='param for sphere')
  parser.add_argument('--beta-min', type=float, default=5., help='param for sphere')
  parser.add_argument('--beta-freeze', type=int, default=0, help='param for sphere')
  parser.add_argument('--gamma', type=float, default=0.12, help='param for sphere')
  parser.add_argument('--power', type=float, default=1.0, help='param for sphere')
  parser.add_argument('--scale', type=float, default=0.9993, help='param for sphere')
  parser.add_argument('--rand-mirror', type=int, default=1, help='if do random mirror in training')
  parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
  parser.add_argument('--target', type=str, default='lfw,cfp_fp,agedb_30', help='verification targets')
  #parser.add_argument('--target', type=str, default= 'lfw', help='verification targets')
  parser.add_argument('--log-file', type=str, default='trainlog', help='the name of log file')
  parser.add_argument('--log-dir', type=str, default='/home/zhongyaoyao/insightface/', help='directory of the log file')
  parser.add_argument('--interweight', type=float, default=0.01, help='param for interloss')
  parser.add_argument('--noise-beta', type=float, default=0.9, help='param for interloss')
  args = parser.parse_args()
  return args


def get_symbol(args, arg_params, aux_params):
  data_shape = (args.image_channel,args.image_h,args.image_w)
  image_shape = ",".join([str(x) for x in data_shape])
  if args.network[0]=='so':
    print('init spherenet_o', args.num_layers)
    embedding = spherenet.get_symbol(0, args.emb_size, args.num_layers)
  elif args.network[0]=='':
    print('init spherenet', args.num_layers)
    embedding = spherenet_bn.get_symbol(args.emb_size, args.num_layers)
  else:
    print('init resnet', args.num_layers)
    embedding = fresnet.get_symbol(args.emb_size, args.num_layers,
        version_se=args.version_se, version_input=args.version_input, 
        version_output=args.version_output, version_unit=args.version_unit,
        version_act=args.version_act)
  nembedding = mx.symbol.L2Normalization(embedding, mode='instance')
  out_list = [mx.symbol.BlockGrad(embedding)]
  all_label = mx.symbol.Variable('softmax_label')

  label_softmax = mx.sym.slice_axis(all_label, axis=0, begin=0, end=args.batch_size // args.ctx_num)
  nembedding_softmax = mx.sym.slice_axis(nembedding, axis=0, begin=0, end=args.batch_size // args.ctx_num)
  label_inter = mx.sym.slice_axis(all_label, axis=0, begin=args.batch_size // args.ctx_num,
                                  end=args.batch_size // args.ctx_num + args.batchsize_id // args.ctx_num)
  nembedding_inter = mx.sym.slice_axis(nembedding, axis=0, begin=args.batch_size // args.ctx_num,
                                       end=args.batch_size // args.ctx_num + args.batchsize_id // args.ctx_num)

  # nembedding_inter = mx.symbol.L2Normalization(embedding_inter, mode='instance')
  nembedding_inter = mx.sym.transpose(nembedding_inter)
  nembedding_inter = mx.symbol.reshape(nembedding_inter, (
  args.emb_size, args.batchsize_id // (args.ctx_num * args.images_per_identity), args.images_per_identity))
  nembedding_inter = mx.sym.transpose(nembedding_inter, axes=(2, 1, 0))  # 3*id*512
  nembedding_inter = mx.sym.mean(nembedding_inter, axis=0)
  nembedding_inter = mx.sym.L2Normalization(nembedding_inter, mode='instance')
  emb_norm = mx.sym.norm(nembedding_inter)
  nembedding_inter_t = mx.sym.transpose(nembedding_inter)
  cosine_matrix = mx.sym.dot(nembedding_inter, nembedding_inter_t)
  cosine_matrix = cosine_matrix - mx.symbol.eye(args.batchsize_id // (args.ctx_num * args.images_per_identity))
  cosine_matrix = cosine_matrix * cosine_matrix
  inter_loss = args.interweight * mx.symbol.mean(cosine_matrix)
  inter_loss = mx.sym.MakeLoss(inter_loss)

  if args.loss_type==0:
      _weight = mx.symbol.Variable('fc7_weight')
      _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
      fc7 = mx.sym.FullyConnected(data=nembedding_softmax, weight=_weight, bias=_bias, num_hidden=args.num_classes, name='fc7')
  else:
      s = args.margin_s
      m = args.margin_m
      assert s > 0.0
      assert m >= 0.0
      assert m < (math.pi / 2)
      _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0,
                                   wd_mult=args.fc7_wd_mult)
      _weight = mx.symbol.L2Normalization(_weight, mode='instance')
      nembedding_softmax = nembedding_softmax * s
      fc7 = mx.sym.FullyConnected(data=nembedding_softmax, weight=_weight, no_bias=True, num_hidden=args.num_classes, name='fc7')
      zy = mx.sym.pick(fc7, label_softmax, axis=1)
      cos_t = zy / s
      cos_m = math.cos(m)
      sin_m = math.sin(m)
      mm = math.sin(math.pi - m) * m
      # threshold = 0.0
      threshold = math.cos(math.pi - m)
      if args.easy_margin:
          cond = mx.symbol.Activation(data=cos_t, act_type='relu')
      else:
          cond_v = cos_t - threshold
          cond = mx.symbol.Activation(data=cond_v, act_type='relu')
      body = cos_t * cos_t
      body = 1.0 - body
      sin_t = mx.sym.sqrt(body)
      new_zy = cos_t * cos_m
      b = sin_t * sin_m
      new_zy = new_zy - b
      new_zy = new_zy * s
      if args.easy_margin:
          zy_keep = zy
      else:
          zy_keep = zy - s * mm
      new_zy = mx.sym.where(cond, new_zy, zy_keep)

      diff = new_zy - zy
      diff = mx.sym.expand_dims(diff, 1)
      gt_one_hot = mx.sym.one_hot(label_softmax, depth=args.num_classes, on_value=1.0, off_value=0.0)
      body = mx.sym.broadcast_mul(gt_one_hot, diff)
      fc7 = fc7 + body

  #1
  #softmaxloss = mx.symbol.SoftmaxOutput(data=fc7, label=label_softmax, name='softmax', normalization='valid')
  #2
  #softmax=mx.sym.softmax_cross_entropy(data=fc7, label=label_softmax)
  #softmaxloss = mx.sym.MakeLoss(softmax)
  #3
  if args.noise:
      softmaxs = mx.sym.log_softmax(data=fc7, name="softmax")
      pred_label = mx.sym.argmax(softmaxs, axis=1)
      pred_one_hot = mx.sym.one_hot(pred_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
      gt_one_hot = mx.sym.one_hot(label_softmax, depth=args.num_classes, on_value=1.0, off_value=0.0)
      cross_entropy_gt = - mx.sym.sum(mx.sym.broadcast_mul(gt_one_hot, softmaxs), axis=[0, 1])
      cross_entropy_pred = - mx.sym.sum(mx.sym.broadcast_mul(pred_one_hot, softmaxs), axis=[0, 1])
      cross_entropy = args.noise_beta*cross_entropy_gt + (1 - args.noise_beta)*cross_entropy_pred
      cross_entropy = cross_entropy / (args.batch_size // 2)
      softmaxloss = mx.sym.MakeLoss(cross_entropy)
  else:
      softmaxs = mx.sym.log_softmax(data=fc7, name="softmax")
      gt_one_hot = mx.sym.one_hot(label_softmax, depth=args.num_classes, on_value=1.0, off_value=0.0)
      cross_entropy = - mx.sym.sum(mx.sym.broadcast_mul(gt_one_hot, softmaxs), axis=[0, 1])
      cross_entropy = cross_entropy/(args.batch_size // 2)
      softmaxloss = mx.sym.MakeLoss(cross_entropy)


  out_list.append(mx.symbol.BlockGrad(label_softmax))
  out_list.append(mx.symbol.BlockGrad(fc7))
  out_list.append(mx.symbol.BlockGrad(label_inter))
  out_list.append(mx.symbol.BlockGrad(softmaxs))
  out_list.append(softmaxloss)
  out_list.append(inter_loss)
  out_list.append(mx.symbol.BlockGrad(emb_norm))
  out_list.append(mx.symbol.BlockGrad(cosine_matrix))
  out = mx.symbol.Group(out_list)
  return (out, arg_params, aux_params)


def train_net(args):
    ctx = []
    cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    if len(cvd)>0:
      for i in xrange(len(cvd.split(','))):
        ctx.append(mx.gpu(i))
    if len(ctx)==0:
      ctx = [mx.cpu()]
      print('use cpu')
    else:
      print('gpu num:', len(ctx))
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
      os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    args.batch_size = args.batch_size1
    args.rescale_threshold = 0
    args.image_channel = 3

    os.environ['BETA'] = str(args.beta)
    data_dir_list = args.data_dir.split(',')
    assert len(data_dir_list)==1
    data_dir = data_dir_list[0]
    prop = face_image.load_property(data_dir)
    args.num_classes = prop.num_classes
    image_size = prop.image_size
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    assert(args.num_classes>0)
    print('num_classes', args.num_classes)
    path_imgrec1 = os.path.join(data_dir, "train.rec")

    data_dir_interclass_list = args.data_dir_interclass.split(',')
    assert len(data_dir_interclass_list)==1
    data_dir_interclass = data_dir_interclass_list[0]
    path_imgrec2 = os.path.join(data_dir_interclass, "train.rec")

    if args.loss_type==1 and args.num_classes>20000:
      args.beta_freeze = 5000
      args.gamma = 0.06

    print('Called with argument:', args)
    data_shape = (args.image_channel,image_size[0],image_size[1])
    mean = None

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained)==0:
      arg_params = None
      aux_params = None
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
      print('sym: ', sym)

    else:
      vec = args.pretrained.split(',')
      #print('loading', vec)
      _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
      #if 'fc7_weight' in arg_params.keys():
      #  del arg_params['fc7_weight']
      sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)

    sym_test, _, _ = get_symbol(args, arg_params, aux_params)
    #label_name = 'softmax_label'
    #label_shape = (args.batch_size,)
    model = mx.mod.Module(
        context       = ctx,
        symbol        = sym,
    )
    val_dataiter = None

    train_dataiter = FaceImageIter(
        mx_model             = model,
        ctx                  = ctx,
        ctx_num              = args.ctx_num,
        data_shape           = data_shape,
        batch_size1          = args.batch_size1,
        path_imgrec1         = path_imgrec1,
        batchsize_id         = args.batchsize_id,
        batch_size2          = args.batch_size2,
        path_imgrec2         = path_imgrec2,
        images_per_identity  = args.images_per_identity,
        interclass_bag_size  = args.bag_size,
        shuffle              = True,
        aug_list             = None,
        rand_mirror          = True,
    )

    eval_metrics = [mx.metric.create(AccMetric()),mx.metric.create(LossValue()),mx.metric.create(LossValue2())]

    if args.network[0]=='r' or args.network[0]=='y':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2) #resnet style
    elif args.network[0]=='i' or args.network[0]=='x':
      initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2) #inception
    else:
      initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0/args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 2
    _cb = mx.callback.Speedometer(args.batch_size, som)

    ver_list = []
    ver_name_list = []
    for name in args.target.split(','):
      path = os.path.join(data_dir,name+".bin")
      #path = os.path.join('/ssd/MegaFace/MF2_aligned_pic9/', name + ".bin")
      if os.path.exists(path):
        data_set = verification.load_bin(path, image_size)
        ver_list.append(data_set)
        ver_name_list.append(name)
        print('ver', name)

    model_t = None


    def ver_test(nbatch, model_t):
      results = []

      if model_t is None:
          all_layers = model.symbol.get_internals()
          symbol_t = all_layers['blockgrad0_output']
          model_t = mx.mod.Module(symbol=symbol_t, context=ctx, label_names=None)
          print([('data', (10,) + data_shape)])
          model_t.bind(data_shapes=[('data', (10,) + data_shape)])
          arg_t, aux_t = model.get_params()
          model_t.set_params(arg_t, aux_t)
      else:
          arg_t, aux_t = model.get_params()
          model_t.set_params(arg_t, aux_t)

      for i in xrange(len(ver_list)):
        acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(ver_list[i], model_t, 10, 10, None, None)
        print('[%s][%d]XNorm: %f' % (ver_name_list[i], nbatch, xnorm))
        #print('[%s][%d]Accuracy: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc1, std1))
        print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (ver_name_list[i], nbatch, acc2, std2))
        results.append(acc2)
      return results



    highest_acc = [0.0, 0.0, 0.0]  #lfw and target
    #for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [0]
    save_step = [0]
    if len(args.lr_steps)==0:
      lr_steps = [100000, 140000, 160000]
      p = 512.0/args.batch_size
      for l in xrange(len(lr_steps)):
        lr_steps[l] = int(lr_steps[l]*p)
    else:
      lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)
    def _batch_callback(param):
      #global global_step
      global_step[0]+=1
      mbatch = global_step[0]
      for _lr in lr_steps:
        if mbatch==args.beta_freeze+_lr:
          opt.lr *= 0.1
          print('lr change to', opt.lr)
          break

      _cb(param)
      #if mbatch%1==0:
        #print('mbatch:',mbatch)
        #arg, aux = model.get_params()


      if mbatch==1:
          ver_test(mbatch, model_t)
          arg, aux = model.get_params()
          mx.model.save_checkpoint(prefix, 1000, model.symbol, arg, aux)
          print('lr-batch-epoch:', opt.lr, param.nbatch, param.epoch)

      if mbatch>=0 and mbatch%args.verbose==0:
        arg, aux = model.get_params()
        mx.model.save_checkpoint(prefix, 0, model.symbol, arg, aux)
        acc_list = ver_test(mbatch,model_t)
        save_step[0]+=1
        msave = save_step[0]
        do_save = False
        if do_save:
          print('saving', msave)
          arg, aux = model.get_params()
          mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
        print('[%d]Accuracy-Highest: %1.5f %1.5f %1.5f'%(mbatch, highest_acc[0], highest_acc[1], highest_acc[2]))
      if mbatch<=args.beta_freeze:
        _beta = args.beta
      else:
        move = max(0, mbatch-args.beta_freeze)
        _beta = max(args.beta_min, args.beta*math.pow(1+args.gamma*move, -1.0*args.power))
      #print('beta', _beta)
      os.environ['BETA'] = str(_beta)
      if args.max_steps>0 and mbatch>args.max_steps:
        sys.exit(0)

    epoch_cb = None

    #print('arg_params',arg_params,aux_params)
    model.fit(train_dataiter,
        begin_epoch        = begin_epoch,
        num_epoch          = end_epoch,
        eval_data          = val_dataiter,
        eval_metric        = eval_metrics,
        kvstore            = 'device',
        optimizer          = opt,
        #optimizer_params   = optimizer_params,
        initializer        = initializer,
        arg_params         = arg_params,
        aux_params         = aux_params,
        allow_missing      = True,
        batch_end_callback = _batch_callback,
        epoch_end_callback = epoch_cb )

def main():
    #time.sleep(3600*6.5)
    global args
    args = parse_args()
    train_net(args)

if __name__ == '__main__':
    main()

