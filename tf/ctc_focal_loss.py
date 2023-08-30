#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Focal CTC Loss."""

import tensorflow as tf


class CTCFocalLoss():
    '''
    Ref: Focal CTC Loss for Chinese Optical Character Recognition on Unbalanced Datasets
        https://downloads.hindawi.com/journals/complexity/2019/9345861.pdf

    p = e^(-ctc_loss)
    focal_loss = alpha*(1-p)^gamma*ctc_loss
    '''

    def __init__(self, alpha=0.5, gamma=0.5, logits_time_major=False, blank_index=-1, lsm_prob=0.0, name="CTCFocalLoss"):
        self.logits_time_major = logits_time_major
        self.blank_index = blank_index
        self.need_logit_length = True
        self.lsm_prob = lsm_prob
        self.gamma = gamma
        self.alpha = alpha
        self.name = name

    def __call__(self, input, label, input_length, labels_length):
        ctc_loss = tf.nn.ctc_loss_v2(
            labels,
            logits,
            label_length,
            logit_length,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index,
            name=self.name
        )

        p = tf.math.exp(-ctc_loss)
        focal_ctc_loss = ((self.alpha)*((1-p)**self.gamma)*(ctc_loss))
        loss = tf.reduce_mean(focal_ctc_loss)

        return loss
