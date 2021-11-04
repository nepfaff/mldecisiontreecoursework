# MLDecisionTreeCoursework

## Project Setup

Please follow the following setup instructions before using any of the functionality provided by this package.

1. Ensure that the directory `mldecisiontreecoursework` is present on an Ubuntu machine.
2. Install `mldecisiontreecoursework` by running `pip3 install ./path/to/mldecisiontreecoursework` where `path/to/mldecisiontreecoursework` is the path to the `mldecisiontreecoursework` directory.

## Producing Evaluation Results

1. Open a terminal inside the `mldecisiontreecoursework` directory.
2. Run one of the following commands:
    - Printing the results to the terminal: `./evaluate_decision_tree_algorithm.py`
    - Printing the results to a specified text file: `./evaluate_decision_tree_algorithm.py --path "path/to/file.txt"` where `path/to/file.txt` is the     path to the text file to append the results to.

This script produces evaluation results for the decision tree algorithm without and with pruning for both the clean and the noisy data sets. These data sets can be found in `mldecisiontreecoursework/Data/intro2ML-coursework1/wifi_db`.

Note, that this script might take a couple of minutes to execute. Moreover, there might be a delay between the output of the clean and the output of the noisy data set. Please don't terminate the script before both results have been produced.

## Running Individual Functions

Instructions for running the most important functionality are given below. Please consult the individual docstrings in the source code for additional information.

### Loading data sets from a text file

The following code snipet shows how to load data sets from a text file. `data_file_path` is the `string` path to the text file containing the data. Each row in the file should contain a number of attributes of type `float` followed by one label of type `int` where individual columns are separated by whitespace. Each row represents one instance. `number_of_attributes` is the `int` number of attributes that the data set contains which is one less than the number of columns. `x` is a `np.ndarray` of type `float` and shape `(n,k)` where `n` is the number of instances and `k` is the number of attributes. `y` is a `np.ndarray` of type `int` and shape `(n,)`.

```python
from data_loading import load_txt_data

x, y = load_txt_data(data_file_path, number_of_attributes)
```

### Training a decision tree without pruning

The following code snipet shows how to train a decision tree using training instances `x_train` and the corresponding class labels `y_train`. `x_train` is a `np.ndarray` of type `float` and shape `(n,k)` where `n` is the number of instances and `k` is the number of attributes. `y_train` is a `np.ndarray` of type `int` and shape `(n,)`. `decision_tree` is a `dictionary` representation of the trained decision tree. `depth` is the `int` depth of the tree.

```python
from decision_tree import decision_tree_learning

decision_tree, depth = decision_tree_learning(x_train, y_train)
```

### Training a decision tree with pruning

The following code snipet shows how to prune a trained decision tree using a separate validation set. `x_train` and the corresponding class labels `y_train`. `x_train` is a `np.ndarray` of type `float` and shape `(n,k)` where `n` is the number of instances and `k` is the number of attributes. `y_train` is a `np.ndarray` of type `int` and shape `(n,)`. `x_validation` and `y_validation` have the same format as `x_train` and `y_train`. `pruned_decision_tree` is a pruned version of `decision_tree` and `validation_errors` is the `int` number of validation errors produced by the pruned tree.

```python
from decision_tree import decision_tree_learning, decision_tree_pruning

decision_tree, depth = decision_tree_learning(x_train, y_train)

pruned_decision_tree, validation_errors = decision_tree_pruning(decision_tree, x_train, y_train, x_validation, y_validation)
```

### Using a trained decision tree to predict labels

The following code snipet shows how to use a trained decision tree (pruned or unpruned) to predict the class labels for instances. `decision_tree` is either a pruned or unpruned decision tree as produced in the above code snipets. `x` is the instances that we want to predict labels for. It is a `np.ndarray` of type `float` and shape `(n,k)` where `n` is the number of instances and `k` is the number of attributes. `y_predict` is the predicted class labels corresponding to `x`. It is a `np.ndarray` of type `int` and shape `(n,)`.

```python
from decision_tree import decision_tree_predict

y_predict = decision_tree_predict(decision_tree, x)
```

### Evaluating the decision tree without pruning algorithm using cross-validation

The following code snipet shows how to evaluate the decision tree algorithm without pruning using cross-validation. `x` is a `np.ndarray` of type `float` and shape `(n,k)` where `n` is the number of instances and `k` is the number of attributes. `y` is a `np.ndarray` of type `int` and shape `(n,)`. `evaluation` is an instance of the `Evaluation` class which contains the evaluation metrics. `confusion_matrix` is an averaged confusion matrix of type `float` and shape `(c,c)`, where `c` is the number of classes. The confusion matrix rows represent the actual classes and the columns the predicted classes. `accuracy` is the averaged accuracy. `precisions` are the averaged precisions per class where the lowest index represents the class with the lowest value. `recalls` are the averaged recalls per class where the lowest index represents the class with the lowest value. `f1s` are the averaged F1-measures per class where the lowest index represents the class with the lowest value.

```python
from evaluation import cross_validation

evaluation = cross_validation(x, y)

# Display individual evaluation metrics:
print(evaluation.confusion_matrix)
print(evaluation.accuracy)
print(evaluation.precisions)
print(evaluation.recalls)
print(evaluation.f1s)
```

### Evaluating the decision tree with pruning algorithm using nested cross-validation

The following code snipet shows how to evaluate the decision tree algorithm with pruning using cross-validation. `x` is a `np.ndarray` of type `float` and shape `(n,k)` where `n` is the number of instances and `k` is the number of attributes. `y` is a `np.ndarray` of type `int` and shape `(n,)`. `evaluation` is an instance of the `Evaluation` class which contains the evaluation metrics. `confusion_matrix` is an averaged confusion matrix of type `float` and shape `(c,c)`, where `c` is the number of classes. The confusion matrix rows represent the actual classes and the columns the predicted classes. `accuracy` is the averaged accuracy. `precisions` are the averaged precisions per class where the lowest index represents the class with the lowest value. `recalls` are the averaged recalls per class where the lowest index represents the class with the lowest value. `f1s` are the averaged F1-measures per class where the lowest index represents the class with the lowest value.

```python
from evaluation import nested_cross_validation

evaluation = nested_cross_validation(x, y)

# Display individual evaluation metrics:
print(evaluation.confusion_matrix)
print(evaluation.accuracy)
print(evaluation.precisions)
print(evaluation.recalls)
print(evaluation.f1s)
```

## Visualisation

To see a visualisation of the trained tree simply run `./visualise_decision_tree`.
A .png file named `clean_unprunned.png` can be found in the `figures/` folder'.

## Testing

The directory `mldecisiontreecoursework/test` contains unit tests for the majority of the functionality present in `mldecisiontreecoursework`. The testing framework used is `pytest`/`pytest-3`. All unit tests can be run using the script `mldecisiontreecoursework/test/run_tests.sh`. For example, from inside the `mldecisiontreecoursework` directory, this script can be called using `./test/run_tests.sh`.
