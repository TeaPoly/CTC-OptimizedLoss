#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2023 Lucky Wong
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
"""Oracle and 1-best Hypothesis CTC loss definition."""

import torch
import torchaudio
from torchaudio.models.decoder import ctc_decoder


class O1Loss(torch.nn.Module):
    """
    O-1: Self-training with Oracle and 1-best Hypothesis
    https://arxiv.org/abs/2308.07486
    """

    def __init__(self, vocab_size, beam_size=8):
        """
        Args:
            beam_size (int): number of best decodings to return
        """
        super().__init__()
        tokens = [str(i) for i in range(vocab_size)]
        self.beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=tokens,
            nbest=beam_size,
            beam_size=beam_size,
            blank_token="0",
            sil_token="1",
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
                O-1 loss.
        """
        beam_search_results = self.beam_search_decoder(
            emissions.cpu(), emissions_lengths.cpu())

        loss = torch.tensor(0.0)
        for batch_idx in range(emissions.size(0)):
            groud_turth = labels[batch_idx, :labels_length[batch_idx]].cpu()
            beam_search_result = beam_search_results[batch_idx]

            # The hypothesis with the best WER serves as the oracle hypothesis Y
            # oracle and the hypothesis with the top probability becomes the 1-best hypothesis Y1−best.
            # Both oracle and 1-best hypotheses are chosen from the beam while dropping the
            # rest of the hypotheses.
            for idx, one_result in enumerate(beam_search_result):
                tokens = one_result.tokens[1:-1]
                score = one_result.score
                beam_search_wer = torchaudio.functional.edit_distance(
                    groud_turth, tokens) / len(groud_turth)
                if idx == 0:
                    # 1-best
                    one_best_score = score
                    one_best_wer = beam_search_wer
                    # orcle
                    oracle_wer = beam_search_wer
                    oracle_score = score
                elif oracle_wer < beam_search_wer:
                    # orcle
                    oracle_wer = beam_search_wer
                    oracle_score = score
            loss_o1 = -oracle_score*(1-oracle_wer)+one_best_score*one_best_wer
            loss += loss_o1
        return loss.to(emissions.device)
