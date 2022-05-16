# ICLR Computational Geometry & Topology Challenge 2022

Welcome to the ICLR 2022 Computational Geometry & Topology challenge 2022 --- by the [ICLR 2022 Workshop on Geometrical and Topological Representation Learning](https://gt-rl.github.io/).

Lead organizers: [Adele Myers](https://ahma2017.wixsite.com/adelemyers), Saiteja Utpala, and Nina Miolane (UC Santa Barbara).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6554617.svg)](https://doi.org/10.5281/zenodo.6554617)


## Description of the challenge

The purpose of this challenge is to foster reproducible research in geometric (deep) learning, by crowdsourcing the open-source implementation of learning algorithms on manifolds. Participants are asked to contribute code for a published/unpublished algorithm, following Scikit-Learn/Geomstats' or pytorch's APIs and computational primitives, benchmark it, and demonstrate its use in real-world scenarios.

Each submission takes the form of a Jupyter Notebook leveraging the coding infrastructure and building blocks from the package [Geomstats](https://github.com/geomstats/geomstats). The participants submit their Jupyter Notebook via [Pull Requests](https://github.com/geomstats/challenge-iclr-2022/pulls) (PR) to this GitHub repository, see [Guidelines](#guidelines) below.

In addition to the challenge's prizes, participants will have the opportunity to co-author a white paper summarizing the findings of the competition.

This is the second edition of this challenge! Feel free to look at [last year's guidelines, submissions](https://github.com/geomstats/challenge-iclr-2021), [winners](https://gt-rl.github.io/2021/challenge) and [paper](https://arxiv.org/abs/2108.09810) for additional information.

**Note:** _We invite participants to review this README regularly, as details are added to the guidelines when questions are submitted to the organizers._

## Deadline

The final Pull Request submission date and hour will have to take place before:
- **April 4th, 2022 at 16:59 PST (Pacific Standard Time)**. 

The participants can freely commit to their Pull Request and modify their submission until this time.

## Winners announcement and prizes

The first 3 winners will be announced at the ICLR 2022 virtual workshop [Geometrical and Topological Representation Learning](https://gt-rl.github.io/) and advertised through the web. The winners will also be contacted directly via email. 

The prizes are:
- $2000 for the 1st place,
- $1000 for the 2nd place,
- $500 for the 3rd place.
 
## Subscription

Anyone can participate and participation is free. It is enough to:
- send a [Pull Request](https://github.com/geomstats/challenge-iclr-2022/pulls),
- follow the challenge [guidelines](#guidelines),
to be automatically considered amongst the participants. 

An acceptable PR automatically subscribes a participant to the challenge.

## Guidelines

We encourage the participants to start submitting their Pull Request early on. This allows to debug the tests and helps to address potential issues with the code.

Teams are accepted and there is no restriction on the number of team members.

The principal developpers of Geomstats (i.e. the co-authors of Geomstats published papers) are not allowed to participate.

A submission should respect the following Jupyter Notebook’s structure:
1. Introduction and Motivation
  - Explain and motivate the choice of learning algorithm
2. Related Work and Implementations
  - Contrast the chosen learning algorithms with other algorithms
  - Describe existing implementations, if any
3. Implementation of the Learning Algorithm --- with guidelines:
  - Follow Scikit-Learn/Geomstats APIs, see [RiemannianKMeans](https://github.com/geomstats/geomstats/blob/d89ee0a4eb8cd178a5de5bccc095fda52d9c0732/geomstats/learning/kmeans.py#L16) example, or Pytorch base classes such as `torch.nn.Module`.
  - IMPORTANT: Use Geomstats computational primitives (e.g. exponential, geodesics, parallel transport, etc). Note that the functions in `geomstats.backend` are not considered computational primitives, as they are only wrappers around autograd, numpy, torch and tensorflow functions.
4. Test on Synthetic Datasets and Benchmark
5. Application to Real-World Datasets

## Examples of possible submissions

- Comparing embedding on trees in hyperbolic plane and variants, e.g. from [Sarkar 2011.](https://homepages.inf.ed.ac.uk/rsarkar/papers/HyperbolicDelaunayFull.pdf)
- Hypothesis testing on manifolds, e.g. from [Osborne et al 2013.](https://www.sciencedirect.com/science/article/pii/S0047259X13000602).
- (Extended/Unscented) Kalman Filters on Lie groups and variants, e.g. from [Bourmaud et al 2013.](https://hal.archives-ouvertes.fr/hal-00903252/document)
- Gaussian Processes on Riemannian Manifolds and variants, e.g. from [Calandra et al 2014.](http://mlg.eng.cam.ac.uk/pub/pdf/CalPetRasDei16.pdf)
- Barycenter Subspace Analysis on Manifolds and variants, e.g. from [Pennec 2016.](https://arxiv.org/abs/1607.02833)
- Curve fitting on manifolds and variants, e.g. from [Gousenbourger et al 2018.](https://sites.uclouvain.be/absil/2018.04)
- Smoothing splines on manifolds, e.g. from [Kim et al 2020.](https://arxiv.org/abs/1801.04978)
- Recurrent models on manifolds and variants, e.g. from [Chakraborty et al 2018.](https://proceedings.neurips.cc/paper/2018/file/7070f9088e456682f0f84f815ebda761-Paper.pdf)
- Geodesic CNNs on manifolds and variants, e.g. from [Masci et al 2018.](https://arxiv.org/pdf/1501.06297.pdf)
- Variational autoencoders on Riemannian manifolds and variants, e.g. from [Miolane et al 2019.](https://arxiv.org/abs/1911.08147)
- Probabilistic Principal Geodesic Analysis and variants, e.g. from [Zhang et al 2019.](https://arxiv.org/pdf/1909.01412.pdf)
- Gauge-equivariant neural networks and variants, e.g. from [Cohen et al 2019.](https://arxiv.org/abs/1902.04615)
- and many more, as long as you implement them using Geomstats computational primitives (e.g. exponential, geodesics, parallel transport, etc).

Before starting your implementation, make sure that the algorithm that you want to contribute is not already in the [learning module of Geomstats](https://github.com/geomstats/geomstats/tree/master/geomstats/learning).

The notebook provided in the `submission-example-*` folders is also an example of submission that can help the participants to design their proposal and to understand how to use/inherit from Scikit-Learning, Geomstats, Pytorch. Note that this example is "naive" on purpose and is only meant to give illustrative templates rather than to provide a meaningful data analysis. More examples on how to use the packages can be found on the GitHub repository of [Geomstats](https://github.com/geomstats/geomstats).

The code should be compatible with Python 3.8 and make an effort to respect the Python style guide [PEP8](https://www.python.org/dev/peps/pep-0008/). The portion of the code using `geomstats` only needs to run with `numpy` or `pytorch` backends. However, it will be appreciated by the reviewers/voters if the code can run in all backends: `numpy`, `autograd`, `tensorflow` and `pytorch`, using geomstats `gs.`, when applicable.

The Jupyter notebooks are automatically tested when a Pull Request is submitted. The tests have to pass. Their running time should not exceed 3 hours, although exceptions can be made by contacting the challenge organizers.

If a dataset is used, the dataset has to be public and referenced. There is no constraint on the data type to be used.

A participant can raise GitHub issues and/or request help or guidance at any time through [Geomstats slack](https://geomstats.slack.com/). The help/guidance will be provided modulo availability of the maintainers.


## Submission procedure

1. Fork this repository to your GitHub.

2. Create a new folder with your team leader's GitHub username in the root folder of the forked repository, in the main branch.

3. Place your submission inside the folder created at step 2, with:
- the unique Jupyter notebook (the file shall end with .ipynb),
- datasets (if needed),
- auxiliary Python files (if needed).

Datasets larger than 10MB shall be directly imported from external URLs or from data sharing platforms such as OpenML.

If your project requires external pip installable libraries that are not amongst Geomstats’ requirements.txt, you can include them at the beginning of your Jupyter notebook, e.g. with:
```
import sys
!{sys.executable} -m pip install numpy scipy torch
```

## Evaluation and ranking

The [Condorcet method](https://en.wikipedia.org/wiki/Condorcet_method) will be used to rank the submissions and decide on the winners. The evaluation criteria will be:
1. How "interesting"/"important"/"useful" is the learning algorithm? Note that this is a subjective evaluation criterion, where the reviewers will evaluate what the implementation of this algorithm brings to the community (regardless of the quality of the code).
2. How readable/clean is the implementation? How well does the submission respect Scikit-Learn/Geomstats/Pytorch's APIs? If applicable: does it run across backends?
3. Is the submission well-written? Does the docstrings help understand the methods?
4. How informative are the tests on synthetic datasets, the benchmarks, and the real-world application?

Note that these criteria do not reward new learning algorithms, nor learning algorithms that outperform the state-of-the-art --- but rather clean code and exhaustive tests that will foster reproducible research in our field.

Selected Geomstats maintainers and collaborators, as well as each team whose submission respects the guidelines, will vote once on Google Form to express their preference for the 3 best submissions according to each criterion. Note that each team gets only one vote, even if there are several participants in the team.

The 3 preferences must all 3 be different: e.g. one cannot select the same Jupyter notebook for both first and second place. Such irregular votes will be discarded. A link to a Google Form will be provided to record the votes. It will be required to insert an email address to identify the voter. The voters will remain secret, only the final ranking will be published.

## Questions?

Feel free to contact us through [GitHub issues on this repository](https://github.com/geomstats/challenge-iclr-2022/issues), on Geomstats repository or through [Geomstats slack](https://geomstats.slack.com/). Alternatively, you can contact Nina Miolane at nmiolane@gmail.com.
