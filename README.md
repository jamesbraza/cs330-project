# cs330-project

[Stanford CS330][1]: Class Project

## Datasets

We are using [New Plant Diseases Dataset][2] from Kaggle, containing 256 x 256 images:

> Image dataset containing different healthy and unhealthy crop leaves.

Here's how to easily download it using the Kaggle API:

```bash
kaggle datasets download -p plant-diseases --unzip vipoooool/new-plant-diseases-dataset
```

## Developers

This project was developed using Python 3.10.

### Getting Started

Here is how to create a virtual environment to work with this repo:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

#### Including Code QA Tooling

We love quality code!  If you do too,
run these commands after creating the environment:

```bash
python -m pip install -r requirements-qa.txt
pre-commit install
```

### Debugging with `tensorboard`

Here is how you kick off `tensorboard`:

```bash
tensorboard --logdir training
```


### UPLOAD TRAINING PROCESS
1.run_TL_Training.sh TO run transfer learning training \\
2.run_fine_tune.sh TO run fine-tuning training 
3.run_choice_net.sh run choicenet

pre_process_dataset.py will take a x_process numpy array and reduce dimention to 256 for input into choicenet

Afterwards, go to its URL: [http://localhost:6006/](http://localhost:6006/).

[1]: https://cs330.stanford.edu/
[2]: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
