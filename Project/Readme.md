# **Fighting Class Imbalance with Textual Inversion**

**Task**

Use Textual Inversion to generate high-quality, synthetic images for underperforming or underrepresented classes in a dataset and utilize them to improve classifier performance.

**Pipeline**

The project pipeline is implemented in a Jupyter Notebook, and the steps include:
- Data preparation
- Training textual inversion embeddings
- Generating synthetic data
- Retraining a classifier
- Evaluation of results

Refer to [Whole pipeline notebook](Whole%20pipeline.ipynb) for step-by-step implementation.

**Dataset**

We used the [Oxford 102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), a dataset of flower images with 102 classes, containing:
- Training set: 1,020 images (10 images per class)
- Validation set: 1,020 images (10 images per class)
- Test set: 6,149 images


Use [Demo](https://colab.research.google.com/drive/1O8DJlM2cDOEWDLV9sLvoHXk4fTeywnxd?usp=sharing) to generate images and compare perfomance

For project details refer to [Project presentation](Project_presentation.pdf)

**Examples from the project:**

*Sword lily*
![Sword lily](images/sword%20lily%2050%20gen.png)
*Canna lily*
![Canna lily](images/canna%20lily%2050%20gen.png)
*Azalea*
![Azalea](images/azalea%2050%20gen.png)
