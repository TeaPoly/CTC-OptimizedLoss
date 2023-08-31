# CTC-OptimizedLoss

Some loss optimized for CTC:

`TensorFlow`
-  MWER (minimum WER) Loss with CTC beam search.
-  Knowledge distillation for CTC loss.
-  KL divergence loss for label smoothing.

`PyTorch`
-  O-1: Self-training with Oracle and 1-best Hypothesis.
-  MWER (minimum WER) Loss with CTC beam search.

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

## Citations

``` bibtex
@misc{prabhavalkar2017minimum,
      title={Minimum Word Error Rate Training for Attention-based Sequence-to-Sequence Models}, 
      author={Rohit Prabhavalkar and Tara N. Sainath and Yonghui Wu and Patrick Nguyen and Zhifeng Chen and Chung-Cheng Chiu and Anjuli Kannan},
      year={2017},
      eprint={1712.01818},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{gao2021distilling,
      title={Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition}, 
      author={Yan Gao and Titouan Parcollet and Nicholas Lane},
      year={2021},
      eprint={2005.09310},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{baskar2023o1,
      title={O-1: Self-training with Oracle and 1-best Hypothesis}, 
      author={Murali Karthick Baskar and Andrew Rosenberg and Bhuvana Ramabhadran and Kartik Audhkhasi},
      year={2023},
      eprint={2308.07486},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
