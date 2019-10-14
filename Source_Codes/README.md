## An Adversarial Domain Adaptation Network for Cross-Domain Fine-Grained Recognition

Here we release the code for < An Adversarial Domain Adaptation Network for Cross-Domain Fine-Grained Recognition > in the semi-supervised setting on two fine-grained datasets and one generic dataset.

### Prerequirments
    Python 3.6 with Numpy supported
    PyTorch 1.0

### Layout
    ./AlexNet
      ./Gsv/                        # codes used on Gsv cars dataset with the backbone of alexnet
      ./Office/                     # codes used on Office dataset with the backbone of alexnet
      ./MegRetail/                  # codes used on MegRetail dataset with the backbone of alexnet
    ./ResNet
      ./Gsv/                        # codes used on Gsv cars dataset with the backbone of resnet
      ./Office/                     # codes used on Office dataset with the backbone of resnet
      ./MegRetail/                  # codes used on MegRetail dataset with the backbone of resnet

### Usage

All the configurations user need change are controlled in `data_loader.py` (dataset path) and `solver.py` (the path of pretrained AlexNet Model).

Hyper-parameters are presented in `main.py`.

    $ python main.py

#### Python scripts
1. `main.py` consist of lots of configurations, e.g., learning rate, weight decay.
2. `models.py` defines the nets and some constraints.
3. `solver.py` defines the solver class, including training and testing.
4. `data_loader.py` is designed for loading datasets.
5. `folder.py` is implemented in some semi-supervised evaluations to split the target dataset.
6. `utils.py` provides some useful methods.
