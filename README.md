## MarioNette | [Webpage](https://people.csail.mit.edu/smirnov/marionette/) | [Paper](https://arxiv.org/abs/2104.14553) | [Video](https://youtu.be/KMrdh8RQCJk)

<img src="https://people.csail.mit.edu/smirnov/marionette/im.png" width="75%" alt="MarioNette" />

**MarioNette: Self-Supervised Sprite Learning**<br>
Dmitriy Smirnov, Michaël Gharbi, Matthew Fisher, Vitor Guizilini, Alexei A. Efros, Justin Solomon<br>
[NeurIPS 2021](https://neurips.cc/Conferences/2021)

### Set-up
To install the neecssary dependencies, run:
```
conda env create -f environment.yml
conda activate MarioNette
```
Also, be sure to execute `export PYTHONPATH=:$PYTHONPATH` prior to running any of the scripts.

### Training
To train a MarioNette model, run:
```
python scripts/train.py --checkpoint_dir out_dir --data data_dir
```
Your dataset should be stored in `data_dir`, with each
input frame named `#.png`. If the images are not 128x128 pixels, specify the
resolution using the `--canvas_size` flag.
Optionally, pass a `--layer_size` flag to specify the
anchor grid resolution, `--num_layers` to specify the number of layers, or
`--num_classes` to specify the size of the spirte dictionary.

To monitor the training, launch a TensorBoard instance with `--logdir out_dir`.

### BibTeX
```
@article{smirnov2021marionette,
  title={{MarioNette}: Self-Supervised Sprite Learning},
  author={Smirnov, Dmitriy and Gharbi, Michael and Fisher, Matthew and Guizilini, Vitor and Efros, Alexei A. and Solomon, Justin},
  year={2021},
  journal={Conference on Neural Information Processing Systems}
}
```
