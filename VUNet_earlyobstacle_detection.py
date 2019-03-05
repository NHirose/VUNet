#!/usr/bin/env python
# -*- coding: utf-8 -*-
# From Seigo Ito topic no doki

import rospy
import message_filters
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
import sys
import cv2
import time
import numpy as np
import math
import numpy

#PIL
from image_converter import decode, encode
import ImageDraw
import Image as PILImage

#chainer
import chainer
from chainer import training
from chainer.training import extensions
from chainer import cuda, Variable
from chainer import function
import chainer.functions as F
import chainer.links as L
from chainer import serializers
from chainer.utils import type_check
from PIL import ImageDraw, ImageFont

#GONet-t
model_file_gen = 'nn_model/featlayer_gen_single.h5'
model_file_dis = 'nn_model/featlayer_dis_single.h5'
model_file_invg = 'nn_model/featlayer_invg_single.h5'
model_file_fl = 'nn_model/classlayer_t.h5'

#SNet
model_file_dec = 'nn_model/SNet_dec.h5'
model_file_enc = 'nn_model/SNet_enc.h5'

#DNet
model_file_decD = 'nn_model/DNet_dec.h5'
model_file_encD = 'nn_model/DNet_enc.h5'

#vector size for GONet
nz = 100

#center of picture
xc = 310
yc = 321

yoffset = 310 
xoffset = 310
xyoffset = 275
XYc = [(xc-xyoffset, yc-xyoffset), (xc+xyoffset, yc+xyoffset)]

# resize parameters
rsizex = 128
rsizey = 128

class CBR(chainer.Chain):
    def __init__(self, ch0, ch1, bn=True, sample='down', activation=F.relu, dropout=False):
        self.bn = bn
        self.activation = activation
        self.dropout = dropout
        layers = {}
        w = chainer.initializers.Normal(0.02)
        if sample=='down':
            layers['c'] = L.Convolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        else:
            layers['c'] = L.Deconvolution2D(ch0, ch1, 4, 2, 1, initialW=w)
        if bn:
            layers['batchnorm'] = L.BatchNormalization(ch1)
        super(CBR, self).__init__(**layers)
        
    def __call__(self, x):
        h = self.c(x)
        if self.bn:
            h = self.batchnorm(h)
        if self.dropout:
            h = F.dropout(h)
        if not self.activation is None:
            h = self.activation(h)
        return h

class Encoder(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super(Encoder, self).__init__(**layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,8):
            hs.append(self['c%d'%i](hs[i-1]))
        return hs

class Decoder(chainer.Chain):
    def __init__(self, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.02)
        #layers['l0'] = L.Linear(514, 512, initialW=w)
        layers['c0'] = CBR(514, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c1'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c2'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c3'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(512, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = CBR(256, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6'] = CBR(128, 64, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c7'] = L.Convolution2D(64, out_ch, 3, 1, 1, initialW=w)
        super(Decoder, self).__init__(**layers)

    def __call__(self, hs, vres, wres):
        z = F.concat((hs[-1],vres,wres), axis=1)
        h = self.c0(z)
        for i in range(1,8):
            if i<7:
                h = self['c%d'%i](h)
            else:
                h = F.tanh(self.c7(h))
        return h

class EncoderD(chainer.Chain):
    def __init__(self, in_ch):
        layers = {}
        w = chainer.initializers.Normal(0.01)
        layers['c0'] = L.Convolution2D(in_ch, 64, 3, 1, 1, initialW=w)
        layers['c1'] = CBR(64, 128, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c2'] = CBR(128, 256, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c3'] = CBR(256, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c4'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c5'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c6'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        layers['c7'] = CBR(512, 512, bn=True, sample='down', activation=F.leaky_relu, dropout=False)
        super(EncoderD, self).__init__(**layers)

    def __call__(self, x):
        hs = [F.leaky_relu(self.c0(x))]
        for i in range(1,8):
            hs.append(self['c%d'%i](hs[i-1]))
        return hs

class DecoderD(chainer.Chain):
    def __init__(self, out_ch):
        layers = {}
        w = chainer.initializers.Normal(0.01)
        layers['c0'] = CBR(512, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c1'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c2'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=True)
        layers['c3'] = CBR(1024, 512, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c4'] = CBR(1024, 256, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c5'] = CBR(512, 128, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c6'] = CBR(256, 64, bn=True, sample='up', activation=F.relu, dropout=False)
        layers['c7'] = L.Convolution2D(128, out_ch, 3, 1, 1, initialW=w)
        super(DecoderD, self).__init__(**layers)

    def __call__(self, hs):
        h = self.c0(hs[-1])
        for i in range(1,8):
            h = F.concat((h, hs[-i-1]), axis=1)
            if i<7:
                h = self['c%d'%i](h)
            else:
                h = F.tanh(self.c7(h))
        return h

class Generator(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(Generator, self).__init__(
            l0z = L.Linear(nz, 8*8*512, initialW=initializer),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, initialW=initializer),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, initialW=initializer),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, initialW=initializer),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, initialW=initializer),
            bn0l = L.BatchNormalization(8*8*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )
        
    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn0l(self.l0z(z))), (z.data.shape[0], 512, 8, 8))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        x = (self.dc4(h))
        return x

class invG(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(invG, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initializer),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=initializer),
            l4l = L.Linear(8*8*512, nz, initialW=initializer),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = F.relu(self.c0(x))
        h = F.relu(self.bn1(self.c1(h)))
        h = F.relu(self.bn2(self.c2(h))) 
        h = F.relu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return l

class ELU(function.Function):
    def __init__(self, alpha=1.0):
        self.alpha = numpy.float32(alpha)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types

        type_check.expect(
            x_type.dtype == numpy.float32,
        )

    def forward_cpu(self, x):
        y = x[0].copy()
        neg_indices = x[0] < 0
        y[neg_indices] = self.alpha * (numpy.exp(y[neg_indices]) - 1)
        return y,

    def forward_gpu(self, x):
        y = cuda.elementwise(
            'T x, T alpha', 'T y',
            'y = x >= 0 ? x : alpha * (exp(x) - 1)', 'elu_fwd')(
                x[0], self.alpha)
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        neg_indices = x[0] < 0
        gx[neg_indices] *= self.alpha * numpy.exp(x[0][neg_indices])
        return gx,

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T x, T gy, T alpha', 'T gx',
            'gx = x >= 0 ? gy : gy * alpha * exp(x)', 'elu_bwd')(
                x[0], gy[0], self.alpha)
        return gx,

def elu(x, alpha=1.0):
    return ELU(alpha=alpha)(x)

class Discriminator(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(Discriminator, self).__init__(
            c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, initialW=initializer),
            c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, initialW=initializer),
            c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, initialW=initializer),
            c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, initialW=initializer),
            l4l = L.Linear(8*8*512, 2, initialW=initializer),
            bn0 = L.BatchNormalization(64),
            bn1 = L.BatchNormalization(128),
            bn2 = L.BatchNormalization(256),
            bn3 = L.BatchNormalization(512),
        )
        
    def __call__(self, x, test=False):
        h = elu(self.c0(x))
        h = elu(self.bn1(self.c1(h)))
        h = elu(self.bn2(self.c2(h))) 
        h = elu(self.bn3(self.c3(h)))
        l = self.l4l(h)
        return h

class FL(chainer.Chain):
    def __init__(self, wscale=0.02):
        initializer = chainer.initializers.Normal(wscale)
        super(FL, self).__init__(
            l_img = L.Linear(3*128*128, 10, initialW=initializer),
            l_dis = L.Linear(512*8*8, 10, initialW=initializer),
            l_fdis = L.Linear(512*8*8, 10, initialW=initializer),
            l_LSTM = L.LSTM(30, 30),
            l_FL = L.Linear(30, 1, initialW=initializer),
            bnfl = L.BatchNormalization(2048*7*7),
        )
    def reset_state(self):
        self.l_LSTM.reset_state()

    def set_state(self):
        self.l_LSTM.set_state()
        
    def __call__(self, img_error, dis_error, dis_output, test=False):
        h = F.reshape(F.absolute(img_error), (img_error.data.shape[0], 3*128*128))
        h = self.l_img(h)
        g = F.reshape(F.absolute(dis_error), (dis_error.data.shape[0], 512*8*8))
        g = self.l_dis(g)
        f = F.reshape(dis_output, (dis_output.data.shape[0], 512*8*8))
        f = self.l_fdis(f)
        con = F.concat((h,g,f), axis=1)
        ls = self.l_LSTM(con)
        ghf = F.sigmoid(self.l_FL(ls))
        return ghf

def callback(msg_1):
    global i
    global j
    #global ini
    global xcgg
    global lcg, lhg
    global vel_res, wvel_res #linear velocity, and angular velocity
    global img_nn_c
    global out_gonet
    global vres, wres
    global xc, yc, xyoffset
    j = j + 1

    if j == 1:
        xkg = xcgg #previous image
        xd = np.zeros((3, 3, 128, 128), dtype=np.float32)
        # resize and crop image for msg_1
        cv2_msg_img = bridge.imgmsg_to_cv2(msg_1)
        cv_imgc = bridge.cv2_to_imgmsg(cv2_msg_img, 'rgb8')
        pil_img = encode(cv_imgc)
        fg_img = PILImage.new('RGBA', pil_img.size, (0, 0, 0, 255))
        draw=ImageDraw.Draw(fg_img)
        draw.ellipse(XYc, fill = (0, 0, 0, 0))
        pil_img.paste(fg_img, (0, 0), fg_img.split()[3])
        img_msg = decode(pil_img)
        cv2_imgd = bridge.imgmsg_to_cv2(img_msg, 'rgb8')

        #crop the image
        cv_cutimg = cv2_imgd[yc-xyoffset:yc+xyoffset, xc-xyoffset:xc+xyoffset]
        cv_cutimg = cv2.transpose(cv_cutimg)
        cv_cutimg = cv2.flip(cv_cutimg,1)
        
        #resize the image 3x128x128
        cv_resize1 = cv2.resize(cv_cutimg,(rsizex, rsizey))
        cv_resizex = cv_resize1.transpose(2, 0, 1)
        in_imgcc1 = np.array([cv_resizex], dtype=np.float32)

        #normalization
        in_img1 = (in_imgcc1 - 128)/128


        for i in range(batchsize): #usually batchsize=1
            img_nn_c[i] = in_img1
            vresc[i][0][0][0] = vel_res
            wresc[i][0][0][0] = wvel_res

        xcg = Variable(cuda.to_gpu(img_nn_c))
        vrescg = Variable(cuda.to_gpu(vresc))
        wrescg = Variable(cuda.to_gpu(wresc))

        #designing the virtual velocity from joypad input vel_ref(linear velocity) and wvel_ref(angular velocity)        
        vres[0][0][0] = 0.5
        if wvel_ref == 0.0:
            wres[0][0][0] = 0.0
        elif wvel_ref != 0.0 and vel_ref == 0.0:
            if wvel_ref > 0.0:
                wres[0][0][0] = 1.0
            else:
                wres[0][0][0] = -1.0
        elif vel_ref < 0.0:
            wres[0][0][0] = 0.0                   
        else:
            rd = 2.5*vel_ref/wvel_ref
            wres[0][0][0] = 0.5/rd
            if wres[0][0][0] > 1.0:
                wres[0][0][0] = 1.0
            elif wres[0][0][0] < -1.0:
                wres[0][0][0] = -1.0
        
        vresg = Variable(cuda.to_gpu(vres))   #virtual linear velocity    
        wresg = Variable(cuda.to_gpu(wres))   #virtual angular velocity

        fl.l_LSTM.h = Variable(lhg) #internal value of LSTM
        fl.l_LSTM.c = Variable(lcg) #internal value of LSTM
        
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            img_gen = gen(invg(xcg))
            dis_real = dis(xcg)
            dis_gen = dis(img_gen)
            output = fl(xcg-img_gen, dis_real-dis_gen, dis_real)
        
        out_gonet[0] = np.reshape(cuda.to_cpu(output.data),(batchsize))
        lhg = fl.l_LSTM.h.data
        lcg = fl.l_LSTM.c.data
        xcgg = xcg #keep current image for the next step

        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf",25)
        stext = 35
        for k in range(4):# 4steps prediction
            if k == 0:
               xkg = xkg
               xcg = xcg
            else:
               xkg = xcg
               xcg = xap

            #future prediction:start
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                if k == 0:
                    x = dec(enc(xkg), vrescg, wrescg)
                else:
                    x = dec(enc(xkg), vresg, wresg)

            genk = F.spatial_transformer_sampler(xkg, x)
            xkp = genk

            with chainer.using_config('train', False), chainer.no_backprop_mode():
                xcf = dec(enc(xcg), vresg, wresg)
                xkf = dec(enc(xkp), vresg, wresg)

            gencf = F.spatial_transformer_sampler(xcg, xcf)
            genkf = F.spatial_transformer_sampler(xkp, xkf)

            indata = F.concat((genkf, gencf), axis=1)
            maskg = Variable(cuda.to_gpu(mask_batch))

            with chainer.using_config('train', False), chainer.no_backprop_mode():
                zL = encD(indata)
                xfLs = decD(zL)

            xfL = F.separate(xfLs, axis=1)

            apf0 = F.reshape(xfL[0],(batchsize,1,128,128))
            apf1 = F.reshape(xfL[1],(batchsize,1,128,128))
            apf2 = F.reshape(xfL[2],(batchsize,1,128,128))
            apf3 = F.reshape(xfL[3],(batchsize,1,128,128))
            mask_kL = F.reshape(xfL[4],(batchsize,1,128,128))
            mask_cL = F.reshape(xfL[5],(batchsize,1,128,128))

            apfLc = F.concat((apf0, apf1), axis=1)
            apfLk = F.concat((apf2, apf3), axis=1)
            genLcx = F.spatial_transformer_sampler(gencf, apfLc+maskg)
            genLkx = F.spatial_transformer_sampler(genkf, apfLk+maskg)

            maskL = F.concat((mask_kL, mask_cL), axis=1)
            mask_softL = F.softmax(maskL, axis=1)
            mask_sepL = F.separate(mask_softL, axis=1)

            mask_knL = F.reshape(mask_sepL[0],(batchsize,1,128,128))
            mask_cnL = F.reshape(mask_sepL[1],(batchsize,1,128,128))

            xap = F.scale(genLcx,mask_cnL,axis=0) + F.scale(genLkx,mask_knL,axis=0) #predicted image

            #GONet
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                img_gen = gen(invg(xap))
                dis_real = dis(xap)
                dis_gen = dis(img_gen)
                output = fl(xap-img_gen, dis_real-dis_gen, dis_real)

            out_gonet[k+1] = np.reshape(cuda.to_cpu(output.data),(batchsize))

            #showing predicted image and traversable probability by GONet
            out_imgc = img_nn_c
            imgb = np.fmin(255.0, np.fmax(0.0, out_imgc*128+128))
            imgc = np.reshape(imgb, (3, 128, 128))
            imgd = imgc.transpose(1, 2, 0)
            imge = imgd.astype(np.uint8)
            imgm = bridge.cv2_to_imgmsg(imge)

            imgg = PILImage.new('RGB',(128,30))
            d = ImageDraw.Draw(imgg)
            d.text((stext,0), str(np.round(out_gonet[0][0],2)), fill=(int(255*(1.0-out_gonet[0][0])), int(255*out_gonet[0][0]), 0), font=font)
            imgtext = decode(imgg)
            imgtextcv = bridge.imgmsg_to_cv2(imgtext, 'bgr8')
            imgtextcvx = imgtextcv.transpose(2, 0, 1)
            imgc = np.concatenate((imgc, imgtextcvx), axis=1)
            #
            imgcc0 = imgc

            for im in range(1):
                out_imgc = cuda.to_cpu(xap.data[im])
                imgb = np.fmin(255.0, np.fmax(0.0, out_imgc*128+128))
                imgc = np.reshape(imgb, (3, 128, 128))
                if k == 0:
                    if im == 0:
                        imgg = PILImage.new('RGB',(128,30))
                        d = ImageDraw.Draw(imgg)
                        d.text((stext,0), str(np.round(out_gonet[k+1][im],2)), fill=(int(255*(1.0-out_gonet[k+1][im])), int(255*out_gonet[k+1][im]), 0), font=font)
                        imgtext = decode(imgg)
                        imgtextcv = bridge.imgmsg_to_cv2(imgtext, 'bgr8')
                        imgtextcvx = imgtextcv.transpose(2, 0, 1)
                        imgc = np.concatenate((imgc, imgtextcvx), axis=1)
                        #
                        imgcc1 = imgc
                elif k == 1:
                    if im == 0:
                        imgg = PILImage.new('RGB',(128,30))
                        d = ImageDraw.Draw(imgg)
                        d.text((stext,0), str(np.round(out_gonet[k+1][im],2)), fill=(int(255*(1.0-out_gonet[k+1][im])), int(255*out_gonet[k+1][im]), 0), font=font)
                        imgtext = decode(imgg)
                        imgtextcv = bridge.imgmsg_to_cv2(imgtext, 'bgr8')
                        imgtextcvx = imgtextcv.transpose(2, 0, 1)
                        imgc = np.concatenate((imgc, imgtextcvx), axis=1)
                        #
                        imgcc2 = imgc
                elif k == 2:
                    if im == 0:
                        imgg = PILImage.new('RGB',(128,30))
                        d = ImageDraw.Draw(imgg)
                        d.text((stext,0), str(np.round(out_gonet[k+1][im],2)), fill=(int(255*(1.0-out_gonet[k+1][im])), int(255*out_gonet[k+1][im]), 0), font=font)
                        imgtext = decode(imgg)
                        imgtextcv = bridge.imgmsg_to_cv2(imgtext, 'bgr8')
                        imgtextcvx = imgtextcv.transpose(2, 0, 1)
                        imgc = np.concatenate((imgc, imgtextcvx), axis=1)
                        #
                        imgcc3 = imgc
                elif k == 3:
                    if im == 0:
                        imgg = PILImage.new('RGB',(128,30))
                        d = ImageDraw.Draw(imgg)
                        d.text((stext,0), str(np.round(out_gonet[k+1][im],2)), fill=(int(255*(1.0-out_gonet[k+1][im])), int(255*out_gonet[k+1][im]), 0), font=font)
                        imgtext = decode(imgg)
                        imgtextcv = bridge.imgmsg_to_cv2(imgtext, 'bgr8')
                        imgtextcvx = imgtextcv.transpose(2, 0, 1)
                        imgc = np.concatenate((imgc, imgtextcvx), axis=1)
                        #
                        imgcc4 = imgc

        imgcc = np.concatenate((imgcc0, imgcc1, imgcc2, imgcc3, imgcc4), axis=2)
        imgdd = imgcc.transpose(1, 2, 0)
        imgee = imgdd.astype(np.uint8)
        imgmm = bridge.cv2_to_imgmsg(imgee)
        image_all.publish(imgmm)
        j = 0



def callback_sub2(msg_sub):
    global vel_res, wvel_res
    vel_res = msg_sub.twist.twist.linear.x
    wvel_res = msg_sub.twist.twist.angular.z

def callback_sub3(msg_sub):
    global vel_ref, wvel_ref
    vel_ref = msg_sub.linear.x
    wvel_ref = msg_sub.angular.z

xp = cuda.cupy
cuda.get_device(0).use()

#SNet
enc = Encoder(in_ch=3)
dec = Decoder(out_ch=2)

#DNet
encD = EncoderD(in_ch=6)
decD = DecoderD(out_ch=6)

#GONet
gen = Generator()
dis = Discriminator()
invg = invG()
fl = FL()

serializers.load_hdf5(model_file_invg, invg)
serializers.load_hdf5(model_file_gen, gen)
serializers.load_hdf5(model_file_dis, dis)
serializers.load_hdf5(model_file_fl, fl)

serializers.load_hdf5(model_file_dec, dec)
serializers.load_hdf5(model_file_enc, enc)

serializers.load_hdf5(model_file_decD, decD)
serializers.load_hdf5(model_file_encD, encD)

gen.to_gpu()
dis.to_gpu()
invg.to_gpu()
fl.to_gpu()

enc.to_gpu()
dec.to_gpu()

encD.to_gpu()
decD.to_gpu()

batchsize = 1

i = 0
j = 0
ini = 0

#initial values
vel_res = 0.0
wvel_res = 0.0
vel_ref = 0.0
wvel_ref = 0.0

#response vector
vres = np.ones((batchsize,1,1,1), dtype=np.float32)
wres = np.ones((batchsize,1,1,1), dtype=np.float32)
vresc = np.ones((batchsize,1,1,1), dtype=np.float32)
wresc = np.ones((batchsize,1,1,1), dtype=np.float32)

#lstm initial value
lh = np.zeros((batchsize,30), dtype=np.float32)
lc = np.zeros((batchsize,30), dtype=np.float32)
lhg = cuda.to_gpu(lh)
lcg = cuda.to_gpu(lc)

#image value
img_nn_c = np.zeros((batchsize,3,128,128), dtype=np.float32)
xcgg = Variable(cuda.to_gpu(img_nn_c))
print xcgg.data.shape

#bias flow
mask_batch = np.zeros((batchsize,2,128,128), dtype=np.float32)
mask_x = np.zeros((1,1,128,128), dtype=np.float32)
mask_y = np.zeros((1,1,128,128), dtype=np.float32)

delta = 2.0/127
for i in range(128):
    for jc in range(128):
        mask_x[0][0][i][jc] = delta*jc - 1
        mask_y[0][0][jc][i] = delta*jc - 1
mask = np.concatenate((mask_x, mask_y), axis=1)


for i in range(batchsize):
    mask_batch[i] = mask

#traversable probability by GONet
out_gonet = np.zeros((5,batchsize), dtype=np.float32)

# main function
if __name__ == '__main__':

    #initialize node
    rospy.init_node('sync_topic', anonymous=True)
    #subscribe of topics
    msg1_sub = rospy.Subscriber('/cv_camera_node1/image1', Image, callback)     #current image
    msg2_sub = rospy.Subscriber('/odom', Odometry, callback_sub2)                  #robot velocity
    msg3_sub = rospy.Subscriber('/cmd_vel_mux/input/teleop', Twist, callback_sub3) #tele-operator's velocity

    #publisher of topics
    image_all = rospy.Publisher('img_all',Image,queue_size=10)

    bridge = CvBridge()
    # waiting callback
    print 'waiting message .....'
    rospy.spin()
