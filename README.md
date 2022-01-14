# ICLR Computational Geometry & Topology Challenge 2022

Welcome to the ICLR Computational Geometry & Topology challenge 2022!

Lead organizers: Saiteja Utpala (UC Santa Barbara) and Nina Miolane (UC Santa Barbara)

## Description of the challenge

The purpose of this challenge is to push forward the fields of computational differential geometry and topology.

All participants will have the opportunity to co-author a white paper summarizing the findings of the challenge.

Each submission takes the form of a Jupyter Notebook leveraging the package [Geomstats](https://github.com/geomstats/geomstats). The participants submit their Jupyter Notebook via [Pull Requests](https://github.com/geomstats/challenge-iclr-2022/pulls) (PR) to this GitHub repository, see [Guidelines](#guidelines) below.

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
- TBD

Here is a non-exhaustive list of possible submissions:
- TBD
Etc.

The notebooks provided in the `submission-example-*` folders are examples of data analysis submissions that can help the participants to design their proposal and to understand how to use the packages. Note that these examples are "naive" on purpose and are only meant to give illustrative notebooks templates rather than to provide a meaningful data analysis. More examples on how to use the packages can be found on the GitHub repository of [Geomstats](https://github.com/geomstats/geomstats).

The code should be compatible with Python 3.8 and make an effort to respect the Python style guide [PEP8](https://www.python.org/dev/peps/pep-0008/). The portion of the code using `geomstats` only needs to run with `numpy` backend, `pytorch` and `tensorflow` backends are not required.

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

If your project requires external pip installable libraries that are not amongst Geomstats’ and Giotto-tda’s requirements.txt, you can include them at the beginning of your Jupyter notebook, e.g. with:
```
import sys
!{sys.executable} -m pip install numpy scipy torch
```

## Evaluation and ranking

The [Condorcet method](https://en.wikipedia.org/wiki/Condorcet_method) will be used to rank the submissions and decide on the winners. The main evaluation criterion is general and encourages participants to be creative: "how does the submission help push forward the fields of computational geometry and topology?".

Selected Geomstats/Giotto-tda maintainers and collaborators, as well as each team whose submission respects the guidelines, will vote once on Google Form to express their preference for the 3 best submissions. Note that each team gets only one vote, even if there are several participants in the team.

The 3 preferences must all 3 be different: e.g. one cannot select the same Jupyter notebook for both first and second place. Such irregular votes will be discarded. A link to a Google Form will be provided to record the votes. It will be required to insert an email address to identify the voter. The voters will remain secret, only the final ranking will be published.

## Questions?

Feel free to contact us through [GitHub issues on this repository](https://github.com/geomstats/challenge-iclr-2022/issues), on Geomstats repository or through [Geomstats slack](https://geomstats.slack.com/). Alternatively, you can contact Nina Miolane at nmiolane@gmail.com.
