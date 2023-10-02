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

## Reproducing the lunar detection paper

In order to reproduce the plots from the paper [Opportunities and limits of lunar
gravitational wave detection](https://arxiv.org/abs/2309.15160), 
install the package as above and then run:

```
python -m gw_landscape.lunar_detectors
```

This will create the two plots in the `plots` folder.

The PSDs for the LBI-SUS and LBI-GND concepts are found here, 
in the `gw_landscape/data` folder.

The PSDs for LGWA and other detectors are found in the GWFish repository,
under [`detector_psd`](https://github.com/janosch314/GWFish/tree/main/GWFish/detector_psd).

