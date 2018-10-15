*Note: While this repository is useful for reproducing results from Häusser et al., please consider using the [salad](https://github.com/domainadaptation/salad) domain adaptation library in the future: https://domainadaptation.org*

# Associative Domain Adaptation in PyTorch

This repository contains an implementation of "Associative Domain Adaptation" [[1]](https://arxiv.org/abs/1708.00938).
Right now, it features the `SVHN -> MNIST` transfer as described in the paper.
The results line up the the ones reported in the paper, even slightly better at `Accuracy: 98.06 % / Error: 1.94 %` on the MNIST Validation set.

This implementation is meant to be minimalistic, for easy adaptation to other projects.

To train a model with standard settings, execute

```
> python train.py
```

Notes: 

- The hyperparameters where loosely inspired by the ones reported in the original publication, but not too much finetuning was necessary to get to this result.
- Note the use of the InstanceNormalization layer, which is similar, but not exactly similar to the reference implementation provided by the authors.

## Reference

Original Paper: [https://arxiv.org/abs/1708.00938](https://arxiv.org/abs/1708.00938)
Official Repo: [https://github.com/haeusser/learning_by_association](https://github.com/haeusser/learning_by_association)

```
@inproceedings{haeusser2017associative,
  title={Associative domain adaptation},
  author={Haeusser, Philip and Frerix, Thomas and Mordvintsev, Alexander and Cremers, Daniel},
  booktitle={International Conference on Computer Vision (ICCV)},
  volume={2},
  number={5},
  pages={6},
  year={2017}
}
```

## Contact

In case of any questions with this repository, either use the issue tracker or [contact me](http://stes.io) directly.
