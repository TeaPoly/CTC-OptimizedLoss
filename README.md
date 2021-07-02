# CTC-OptimizedLoss

Some loss optimized for CTC:
-  MWER (minimum WER) Loss with CTC beam search.
-  Knowledge distillation for CTC loss.
-  KL divergence loss for label smoothing.

# Example
```python3
weight = 0.01 # interpolation weight
beam_width = 8 # N-best

mwer_loss = CTCMWERLoss(beam_width=beam_width)(
    ctc_logits, ctc_labels, ctc_label_length, logit_length)

ctc_loss = CTCLoss()(
    ctc_logits, ctc_labels, ctc_label_length, logit_length)

loss = mwer_loss + weight*ctc_loss

```

# Reference
- MINIMUM WORD ERROR RATE TRAINING FOR ATTENTION-BASED SEQUENCE-TO-SEQUENCE MODELS https://arxiv.org/pdf/1712.01818.pdf
- Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition. https://arxiv.org/abs/2005.09310
