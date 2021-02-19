# CTC-MWER
Computes the MWER (minimum WER) Loss with CTC beam search.

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
MINIMUM WORD ERROR RATE TRAINING FOR ATTENTION-BASED SEQUENCE-TO-SEQUENCE MODELS
https://arxiv.org/pdf/1712.01818.pdf
