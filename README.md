# U-Net Tutorial

You can open these notebooks on google colab with GPU support. Just click on the badge below.

Part 1:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osbm/unet_explainer/blob/main/tutorial-part1.ipynb)

Part 1 Solutions:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osbm/unet_explainer/blob/main/tutorial-part1-solutions.ipynb)

Part 2:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osbm/unet_explainer/blob/main/tutorial-part2.ipynb)

Part 2 Solutions:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osbm/unet_explainer/blob/main/tutorial-part2-solutions.ipynb)


## Note if you want to run it locally

First clone the repository:

```
git clone https://github.com/osbm/unet_explainer.git
cd unet_explainer
```

And open a virtual python environment and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

And install the package

```
pip install .
```

Then you can run the notebooks with jupyter in browser or in VSCode:

```
jupyter notebook
```


# Citation for the dataset
```bibtex
@article{ADAMS2022105817,
    title = {Prostate158 - An expert-annotated 3T MRI dataset and algorithm for prostate cancer detection},
    journal = {Computers in Biology and Medicine},
    volume = {148},
    pages = {105817},
    year = {2022},
    issn = {0010-4825},
    doi = {https://doi.org/10.1016/j.compbiomed.2022.105817},
    url = {https://www.sciencedirect.com/science/article/pii/S0010482522005789},
    author = {Lisa C. Adams and Marcus R. Makowski and Günther Engel and Maximilian Rattunde and Felix Busch and Patrick Asbach and Stefan M. Niehues and Shankeeth Vinayahalingam and Bram {van Ginneken} and Geert Litjens and Keno K. Bressem},
    keywords = {Prostate cancer, Deep learning, Machine learning, Artificial intelligence, Magnetic resonance imaging, Biparametric prostate MRI}
}
```