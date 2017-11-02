import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI
import theano.tensor as TT
import theano
import numpy as np
class AttLayer(L.MergeLayer):
    def __init__(self, incoming, W=LI.Normal(0.01), **kwargs):
        super(AttLayer, self).__init__(incoming, **kwargs)
        l_box, l_feat = incoming
        batch_size, num_boxes, feat_size = l_feat.output_shape

        # def add_param(self, spec, shape, name=None, **tags):
        #     if name is not None:
        #         if self.name is not None:
        #             name = "%s.%s" % (self.name, name)
        #     # create shared variable, or pass through given variable/expression
        #     param = utils.create_param(spec, shape, name)
        #     # parameters should be trainable and regularizable by default
        #     tags['trainable'] = tags.get('trainable', True)
        #     tags['regularizable'] = tags.get('regularizable', True)
        #     self.params[param] = set(tag for tag, value in tags.items() if value)
        #     return param
        init_w = np.reshape(np.load('mugfeats.npy'), (1, feat_size))*10
        self.W = self.add_param( init_w, (1, feat_size), name='W')
        self.num_boxes = num_boxes
        self.feat_size = feat_size

    def get_output_for(self, input, **kwargs):
        # these are theano tensors
        boxes, feats = input
        b_size, num_boxes, feat_size = TT.shape(feats)
        tiled_W = TT.tile(self.W,(b_size,1))

        # Rfeats = TT.reshape(feats, (-1, feat_size))
        boxes = TT.reshape(boxes, [-1, self.num_boxes, 4])
        reshaped_W = TT.reshape(tiled_W, (-1, feat_size, 1))
        cos_sim = abs(TT.batched_dot(feats,reshaped_W))
        temp = 1
        exp = TT.reshape(TT.exp(cos_sim),[-1, self.num_boxes])
        Z = TT.tile(TT.sum(exp,axis=1, keepdims=True),[1, self.num_boxes])
        prob1 = TT.reshape(exp/Z, [-1, self.num_boxes, 1])
        prob = TT.tile(prob1, [1,1,4])
        arg_box = TT.sum(prob*boxes,axis=1)
        return arg_box

    def get_output_shape_for(self, input_shapes):
        box_shape, feat_shape = input_shapes
        return (box_shape[0],4)
