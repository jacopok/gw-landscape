# gw-landscape
A plot of the gravitational wave landscape, inspired by [gwplotter](http://gwplotter.com/)

![Plot](plots/landscape.png)

To make it, install the project with [`poetry`](https://python-poetry.org/docs/#installation)
and create a virtual environment:

```
poetry install
poetry shell
```

Alternatively, you can also install the package with 

```
pip install .
```

from the main directory.

Then, run 

```
python -m gw_landscape.plot
```