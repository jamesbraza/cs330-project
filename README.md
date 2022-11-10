# cs330-project

[Stanford CS330][1]: Class Project

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

Afterwards, go to its URL: [http://localhost:6006/](http://localhost:6006/).

[1]: https://cs330.stanford.edu/
