#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2021 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""MWER (minimum WER) Loss definition."""

import tensorflow as tf

def dense_to_sparse(labels, label_seqences):
    '''
    Converts a dense tensor into a sparse tensor.
    
    Args:
        labels: An int Tensor to be converted to a Sparse.
        label_seqences: An list. It is part of the target label that signifies the end of a sentence.
    '''
    mask = tf.sequence_mask(label_seqences, maxlen=tf.shape(labels)[1], dtype=tf.float32)
    indices = tf.where(tf.not_equal(mask, 0))
    return tf.SparseTensor(indices,
            tf.gather_nd(labels, indices),
            tf.shape(labels, out_type=tf.int64)
            )

class CTCMWERLoss():
    """ Computes the MWER (minimum WER) Loss.
        Reference:
        MINIMUM WORD ERROR RATE TRAINING FOR ATTENTION-BASED
        SEQUENCE-TO-SEQUENCE MODELS
        Rohit Prabhavalkar Tara N. Sainath Yonghui Wu Patrick Nguyen
        Zhifeng Chen Chung-Cheng Chiu Anjuli Kannan
        https://arxiv.org/pdf/1712.01818.pdf
    """

    def __init__(self, beam_width=8, name="CTCMWERLoss"):
        """
        Args:
          beam_width: An int scalar >= 0 (beam search beam width).
        """
        self.beam_width = beam_width
        self.top_paths = beam_width

    def loss(self, nbest_decoded, sparse_labels, nbest_log_pdf):

        def word_error_number(hypothesis, truth):
            """
            Computes the Levenshtein distance between sequences to 
            get number of word errors.
            Args:
              hypothesis: A `SparseTensor` containing hypothesis sequences.
              truth: A `SparseTensor` containing truth sequences.
            Returns:
              A dense `Tensor` with rank `R - 1`, where R is the rank of the
              `SparseTensor` inputs `hypothesis` and `truth`.
            Raises:
              TypeError: If either `hypothesis` or `truth` are not a `SparseTensor`.
            """
            return tf.edit_distance(
                hypothesis=hypothesis,
                truth=truth,
                normalize=False)

        # Computes log distribution.
        # log(sum(exp(elements across dimensions of a tensor)))
        sum_nbest_log_pdf = tf.math.reduce_logsumexp(nbest_log_pdf, axis=0) # (batch_size)
        # Re-normalized over just the N-best hypotheses.
        normal_nbest_pdf = [
            tf.exp(log_pdf-sum_nbest_log_pdf) for log_pdf in nbest_log_pdf] # (nbest, batch_size)

        # Number of word errors, but it represents by float.
        nbest_wen = [word_error_number(tf.cast(nbest_decoded[k], dtype=tf.int32), sparse_labels)
                     for k in range(self.top_paths)]  # tf.float32
        # Average number of word errors over the N-best hypohtheses
        mean_wen = tf.reduce_mean(nbest_wen, axis=0)  # tf.float32

        # Re-normalized error word number over just the N-best hypotheses
        normal_nbest_wen = [nbest_wen[k] -
                            mean_wen for k in range(self.top_paths)]

        # Expected number of word errors over the training set.
        mwer_loss = tf.math.reduce_sum(
            [normal_nbest_pdf[k]*normal_nbest_wen[k]
                for k in range(self.top_paths)], axis=0
        ) # compute the sum of loss reduce on nbest

        return tf.math.reduce_mean(mwer_loss) # Computes the mean of mwer loss reduce on batch

    def __call__(self, logit, labels, label_length, logit_length):
        """
        Args:
          labels: tensor of shape [batch_size, max_label_len]
          logits: tensor of shape [batch_size, max_seq_len, vocal_size].
          logit_length: tensor of shape [batch_size] Length of input sequence in
            logits.
          label_length: tensor of shape [batch_size] Length of labels sequence.
        Returns:
          loss: tensor, MWER loss.
        Raises:
          TypeError: if labels is not a SparseTensor.
        """
        time_major_logit = tf.transpose(logit, (1, 0, 2))
        sparse_labels = dense_to_sparse(labels, label_length)

        # Beam search for top-N hypotheses.
        #   decoded: A list of length top_paths.
        #   log_probabilities: A float matrix [batch_size, top_paths]
        #                     containing sequence log-probabilities.
        # NOTE: `nbest_decoded` from CTC Beam search can be generated offline to speed up training.
        nbest_decoded, log_probabilities = tf.nn.ctc_beam_search_decoder(
            time_major_logit, logit_length,
            beam_width=self.beam_width,
            top_paths=self.top_paths)
        nbest_log_pdf = [
            log_probabilities[:, k]
            for k in range(self.top_paths)] # (nbest, batch_size)

        return self.loss(nbest_decoded, sparse_labels, nbest_log_pdf)
