#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Compute KL divergence loss for label smoothing of CTC. 
Modify from https://github.com/hirofumi0810/neural_sp/blob/000cd9dd657f83cd4883faf9ac48d0fcc40badb9/neural_sp/models/criterion.py#L110"""

import tensorflow as tf

def ctc_label_smoothing_loss(logits, ylens):
    """Compute KL divergence loss for label smoothing of CTC models.

    Args:
        logits (Tensor): `[B, T, vocab]`
        ylens (Tensor): `[B]`
    Returns:
        loss_mean (Tensor): `[1]`

    """
    bs, max_ylens, vocab = shape_list(logits)

    log_uniform = tf.zeros_like(logits) + tf.math.log(1/(vocab-1))
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    loss = tf.math.multiply(probs, log_probs - log_uniform)
    ylens_mask = tf.sequence_mask(
        ylens, maxlen=max_ylens, dtype=logits.dtype)
    loss = tf.math.reduce_sum(loss, axis=-1)
    loss_mean = tf.math.reduce_sum(
        loss*ylens_mask)/tf.cast(tf.math.reduce_sum(ylens), dtype=logits.dtype)
    return loss_mean
