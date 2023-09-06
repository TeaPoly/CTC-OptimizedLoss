#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""k2 based ctc loss implementation."""

from typeguard import check_argument_types

import k2
import torch


class K2CTCLoss(torch.nn.Module):
    """K2CTCLoss module."""

    def __init__(
        self,
        odim: int,
        output_beam: int = 10,
        delay_penalty: float = 0.0,  # 0.1
        reduction: str = "none",
        use_double_scores: bool = True,
        modified: bool = False,
        subsampling_factor: int = -1,
    ) -> None:
        """K2CTCLoss module.

        Args:
          odim:
            Number of classes (including blank).
          reduction:
            Specifies the reduction to apply to the output. Default: 'sum'
          device:
            An instance of `torch.device`. Default: 'cpu'
        """
        assert check_argument_types()
        super().__init__()
        self.output_beam = output_beam
        self.delay_penalty = delay_penalty
        self.reduction = reduction
        self.use_double_scores = use_double_scores
        self.modified = modified
        self.subsampling_factor = subsampling_factor

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """k2 ctc forward function.

        Args:
          log_probs:
            Tensor of size (N, T, C), where N is batch size, T is input length,
            and C is number of classes (including blank). Note that it must be
            log probabilities.
          targets:
            Tensor of size (sum(target_lengths)). It represent corresponding labels
            of log_probs.
          input_lengths:
            Tuple or tensor of size (N). It represent the lengths of the inputs.
          target_lengths:
            Tuple or tensor of size (N). It represent lengths of the targets.
        """
        supervision_segments = torch.stack(
            (
                torch.tensor(range(input_lengths.shape[0])),
                torch.zeros(input_lengths.shape[0]),
                input_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)

        if self.subsampling_factor > 1:
            dense_fsa_vec = k2.DenseFsaVec(
                log_probs,
                supervision_segments,
                allow_truncate=self.subsampling_factor - 1,
            )
        else:
            dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

        token_ids = []
        for i in range(target_lengths.size(0)):
            token_ids.append(targets[i, : target_lengths[i]].cpu().tolist())

        decoding_graph = k2.ctc_graph(
            token_ids, modified=self.modified, device=log_probs.device
        )

        ctc_loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=self.output_beam,
            delay_penalty=self.delay_penalty,
            reduction=self.reduction,
            use_double_scores=self.use_double_scores,
        )

        return ctc_loss
