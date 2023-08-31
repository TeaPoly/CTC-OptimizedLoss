#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""CTC based Minimum Word Error Rate Loss definition."""

from typing import Literal

try:
    import k2
except ImportError:
    raise ImportError("You should install K2 to use K2CTC")

import torch


def get_lattice(
    nnet_output: torch.Tensor,
    decoding_graph: k2.Fsa,
    supervision_segments: torch.Tensor,
    search_beam: float,
    output_beam: float,
    min_active_states: int,
    max_active_states: int,
    subsampling_factor: int = 1,
) -> k2.Fsa:
    """Get the decoding lattice from a decoding graph and neural
    network output.
    Args:
      nnet_output:
        It is the output of a neural model of shape `(N, T, C)`.
      decoding_graph:
        An Fsa, the decoding graph. It can be either an HLG
        (see `compile_HLG.py`) or an H (see `k2.ctc_topo`).
      supervision_segments:
        A 2-D **CPU** tensor of dtype `torch.int32` with 3 columns.
        Each row contains information for a supervision segment. Column 0
        is the `sequence_index` indicating which sequence this segment
        comes from; column 1 specifies the `start_frame` of this segment
        within the sequence; column 2 contains the `duration` of this
        segment.
      search_beam:
        Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
        (less pruning). This is the default value; it may be modified by
        `min_active_states` and `max_active_states`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      min_active_states:
        Minimum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to have fewer than this number active.
        Set it to zero if there is no constraint.
      max_active_states:
        Maximum number of FSA states that are allowed to be active on any given
        frame for any given intersection/composition task. This is advisory,
        in that it will try not to exceed that but may not always succeed.
        You can use a very large number if no constraint is needed.
      subsampling_factor:
        The subsampling factor of the model.
    Returns:
      An FsaVec containing the decoding result. It has axes [utt][state][arc].
    """
    dense_fsa_vec = k2.DenseFsaVec(
        nnet_output,
        supervision_segments,
        allow_truncate=subsampling_factor - 1,
    )

    lattice = k2.intersect_dense_pruned(
        decoding_graph,
        dense_fsa_vec,
        search_beam=search_beam,
        output_beam=output_beam,
        min_active_states=min_active_states,
        max_active_states=max_active_states,
    )

    return lattice


class MWERLoss(torch.nn.Module):
    '''Minimum Word Error Rate Loss compuration in k2.

    See equation 2 of https://arxiv.org/pdf/2106.02302.pdf about its definition.
    '''

    def __init__(
        self,
        vocab_size: int,
        subsampling_factor: int,
        search_beam: int = 20,
        output_beam: int = 8,
        min_active_states: int = 30,
        max_active_states: int = 10000,
        temperature: float = 1.0,
        num_paths: int = 100,
        use_double_scores: bool = True,
        nbest_scale: float = 0.5,
        reduction: Literal['none', 'mean', 'sum'] = 'sum'
    ) -> None:
        """
        Args:
          search_beam:
            Decoding beam, e.g. 20.  Smaller is faster, larger is more exact
            (less pruning). This is the default value; it may be modified by
            `min_active_states` and `max_active_states`.
          output_beam:
             Beam to prune output, similar to lattice-beam in Kaldi.  Relative
             to best path of output.
          min_active_states:
            Minimum number of FSA states that are allowed to be active on any given
            frame for any given intersection/composition task. This is advisory,
            in that it will try not to have fewer than this number active.
            Set it to zero if there is no constraint.
          max_active_states:
            Maximum number of FSA states that are allowed to be active on any given
            frame for any given intersection/composition task. This is advisory,
            in that it will try not to exceed that but may not always succeed.
            You can use a very large number if no constraint is needed.
          subsampling_factor:
            The subsampling factor of the model.
          temperature:
            For long utterances, the dynamic range of scores will be too large
            and the posteriors will be mostly 0 or 1.
            To prevent this it might be a good idea to have an extra argument
            that functions like a temperature.
            We scale the logprobs by before doing the normalization.
          use_double_scores:
            True to use double precision floating point.
            False to use single precision.
          reduction:
            Specifies the reduction to apply to the output:
            'none' | 'sum' | 'mean'.
            'none': no reduction will be applied.
                    The returned 'loss' is a k2.RaggedTensor, with
                    loss.tot_size(0) == batch_size.
                    loss.tot_size(1) == total_num_paths_of_current_batch
                    If you want the MWER loss for each utterance, just do:
                    `loss_per_utt = loss.sum()`
                    Then loss_per_utt.shape[0] should be batch_size.
                    See more example usages in 'k2/python/tests/mwer_test.py'
            'sum': sum loss of each path over the whole batch together.
            'mean': divide above 'sum' by total num paths over the whole batch.
          nbest_scale:
            Scale `lattice.score` before passing it to :func:`k2.random_paths`.
            A smaller value leads to more unique paths at the risk of being not
            to sample the path with the best score.
          num_paths:
            Number of paths to **sample** from the lattice
            using :func:`k2.random_paths`.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.search_beam = search_beam
        self.output_beam = output_beam
        self.min_active_states = min_active_states
        self.max_active_states = max_active_states

        self.num_paths = num_paths
        self.nbest_scale = nbest_scale
        self.subsampling_factor = subsampling_factor

        self.mwer_loss = k2.MWERLoss(
            temperature=temperature,
            use_double_scores=use_double_scores,
            reduction=reduction
        )

    def forward(
        self,
        emissions: torch.Tensor,
        emissions_lengths: torch.Tensor,
        labels: torch.Tensor,
        labels_length: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            emissions (torch.FloatTensor): CPU tensor of shape `(batch, frame, num_tokens)` storing sequences of
                probability distribution over labels; output of acoustic model.
            labels (torch.FloatTensor): CPU tensor of shape `(batch, label_len)` storing labels.
            emissions_lengths (Tensor or None, optional): CPU tensor of shape `(batch, )` storing the valid length of
                in time axis of the output Tensor in each batch.
            labels_length (Tensor or None, optional): CPU tensor of shape `(batch, )` storing the valid length of
                label in each batch.

        Returns:
            torch.FloatTensor:
                Minimum Word Error Rate loss.
        """
        H = k2.ctc_topo(
            max_token=self.vocab_size-1,
            modified=False,
            device=emissions.device,
        )

        supervision_segments = torch.stack(
            (
                torch.tensor(range(emissions_lengths.shape[0])),
                torch.zeros(emissions_lengths.shape[0]),
                emissions_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)

        lattice = get_lattice(
            nnet_output=emissions,
            decoding_graph=H,
            supervision_segments=supervision_segments,
            search_beam=self.search_beam,
            output_beam=self.output_beam,
            min_active_states=self.min_active_states,
            max_active_states=self.max_active_states,
            subsampling_factor=self.subsampling_factor,
        )

        token_ids = []
        for i in range(labels_length.size(0)):
            token_ids.append(labels[i, : labels_length[i]].cpu().tolist())

        loss = self.mwer_loss(
            lattice, token_ids,
            nbest_scale=self.nbest_scale,
            num_paths=self.num_paths
        )

        return loss
