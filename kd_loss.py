#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Knowledge distillation for CTC loss definition."""

import tensorflow as tf

class CtcKdLoss():
    """Knowledge distillation for CTC loss.
    Reference
    ---------
    Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition.
    https://arxiv.org/abs/2005.09310
    """

    def __init__(self, logits_time_major=False):
        self.logits_time_major = logits_time_major

    def __call__(self, logits, targets_logits, logit_length, target_logit_length):
        """
        Args:
          targets_logits: tensor of shape [batch_size, max_seq_len, vocal_size], teacher model logits.
          logits: tensor of shape [batch_size, max_seq_len, vocal_size], student model logits.
          logit_length: tensor of shape [batch_size] Length of input sequence in
            logits.
          target_logit_length: tensor of shape [batch_size] Length of teacher model input sequence in
            logits.
        Returns:
          loss: tensor, CTC knowledge distillation loss.
        """
        if not self.logits_time_major:
            targets_logits = tf.transpose(targets_logits, (1, 0, 2))
        decoded, _ = tf.nn.ctc_greedy_decoder(
            targets_logits, target_logit_length)
        ctc_kd_loss = tf.nn.ctc_loss(
            labels=decoded[0],
            inputs=logits,
            sequence_length=logit_length,
            ignore_longer_outputs_than_inputs=True,
            time_major=self.logits_time_major,
            preprocess_collapse_repeated=False
        )

        return tf.reduce_mean(ctc_kd_loss)
