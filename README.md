# Machine Learning Engineer Nanodegree

## Capstone Project - House Prices: Advanced Regression Techniques Using Ensemble Method

German Rezzonico

### Abstract
An ensemble averaging model, based on real estate data, will be used to make predictions about a house’s monetary value. After an exhaustive features engineering, multiple machine learning regression algorithms will be trained and tuned, to obtain a better predictive performance. A model like the one described, could be invaluable for someone like a real estate agent or for one of the many companies operating in the multi billion dollar real estate industry.

### Keywords
Python, regression techniques, feature engineering, grid search, ensemble

### Requirements
This project was developed using **Python 2.7.6** on Chromebook Acer C720 with Ubuntu 14.04.5 LTS installed via [crouton](https://github.com/dnschneid/crouton). The following Python libraries are required:

- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [matplotlib](http://matplotlib.org/)
- [Seaborn](http://seaborn.pydata.org/)
- [Pandas](http://pandas.pydata.org/)
- [sklearn](http://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
- [Pickle](https://docs.python.org/2/library/pickle.html)

It is also required to have [Jupyter Notebook](http://jupyter.org/) installed to run and execute the code.

### Instructions:

#### Install python 2.7, python-pip and python-dev

```sh
sudo apt-get install python2.7 python-pip python-dev
```

##### To verify that you have python and pip installed:

```sh
python --version
pip --version
```

#### Installing Ipython

```sh
sudo apt-get install ipython ipython-notebook
```

#### Installing Jupyter Notebook
-H, --set-home
Request that the security policy set the HOME environment variable to the home directory specified by the target user's password database entry. Depending on the policy, this may be the default behavior.

```sh
sudo -H pip install jupyter
```

##### If any error --> Upgrade pip to the latest version

```sh
sudo -H pip install --upgrade pip
```

##### Try installing Jupyter again

```sh
sudo -H pip install jupyter
```

#### Running Jupyter Notebook

```sh
jupyter notebook
```

#### Install packages

```sh
sudo apt-get install python-numpy python-pandas python-matplotlib


sudo apt-get install build-essential python-dev python-setuptools python-numpy python-scipy libatlas-dev libatlas3gf-base libfreetype6-dev libpng-dev g++ python-matplotlib

sudo apt-get install python-numpy-dev g++

sudo -H pip install pandas scikit-learn

sudo -H pip install seaborn

```

#### Using Jupyter Notebook

Automatically, Jupyter Notebook will show all of the files and folders in the directory it is run from.

To create a new notebook file, select New > Python 2 from the top right pull-down menu.

![Alt jupyter new file](https://assets.digitalocean.com/articles/jupyter_notebook/create_python2_notebook.png)

This will open a notebook. We can now run Python code in the cell or change the cell to markdown. For example, change the first cell to accept Markdown by clicking Cell > Cell Type > Markdown from the top navigation bar. We can now write notes using Markdown and even include equations written in LaTeX by putting them between the $$ symbols. For example, type the following into the cell after changing it to markdown:

```
# Simple Equation

Let us now implement the following equation:
$$ y = x^2$$

where $x = 2$
```

To turn the markdown into rich text, press CTRL+ENTER:

![Alt jupyter markdown](https://assets.digitalocean.com/articles/jupyter_notebook/markdown_results.png)

You can use the markdown cells to make notes and document your code. Let's implement that simple equation and print the result. Select Insert > Insert Cell Below to insert and cell and enter the following code:

```py
x = 2
y = x*x
print y
```

![Alt jupyter run code](https://assets.digitalocean.com/articles/jupyter_notebook/equations_results.png)

#### Datasets

This project uses the [Kaggle version](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) of the [Ames dataset](http://www.amstat.org/publications/jse/v19n3/decock.pdf) compiled by Dean De Cock, which is is a modernized and expanded version of the often cited Boston Housing dataset.

#### Source Code:

##### 01_EDA.ipynb

Contains all the tasks performed to get a better understanding of the data and to obtain all the necessary insights about the data.

##### 3.2.2. 02_features_engineering.ipynb
Here we will be working on three sets of features:

- The two most correlated features with all the missing values fill with zeros (top_2)

- The ten most correlated features with all the missing values fill with zeros (top_10)

- All the engineered features (all)

In each of the three cases the dataset corresponding to: the features (features) of the train set, the logarithmically scaled price (log_price) of ‘SalePrice’ feature of the train set and the public features (public_features) used to make the predictions to be submitted to Kaggle are exported to a .pkl file using Pickle.

##### 3.2.3. 03_default_regression_models.ipynb

The features.pkl, log_price.pkl and public_features.pkl files are loaded again using pickle.
Then the train dataset is divided into a training and testing set, using ‘train_test_split’ from sklearn.cross_validation and the regressors are trained. The R2 score and the RSMLE score of each of the regressors calculated.
The trained regressors are then saved in a regs_dict.dict file using pickle.
The structure of this dictionary is as follow:
```js
regs_dict = { 'top_2': {'untuned': {}, 'tuned': {} }, 'top_10': {'untuned': {}, 'tuned': {} }, 'all': {'untuned': {}, 'tuned': {} } }
```

##### 3.2.4. 04_grid_search.ipynb

Several parameters for each of the regressors are fed for Hyperparameter Optimization. The tuned regressors is then saved in the regs_dict.dict file using pickle.

##### 3.2.5. 05_ensemble_generation.ipynb

The ensemble class is defined and the selected models from the dictionary loaded into the ensemble. Then the R2 and the RMSLE of the ensemble calculated.
The final step consists in calculated the predictions on the public test set and export those results into a .csv file for the corresponding kaggle submission.
