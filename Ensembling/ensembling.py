import numpy as np
import matplotlib
import matplotlib.pyplot as plt
 
import pandas as pd
import math
from collections import deque
from sklearn.model_selection import train_test_split
from time import time
 
 
class Dataset(object):
   
    def __init__(self, df, ix=None):
        """
        :param df: Pandas DataFrame or another Dataset instance. In the latter case only meta data are copied.
        :param ix: boolean index describing samples selected from the original dataset
        """
        if isinstance(df, pd.DataFrame):
            self.columns = list(df.columns)
            self.cdict = {c: i for i, c in enumerate(df.columns)}
            self.data = [df[c].values for c in self.columns]
        elif isinstance(df, Dataset):
            self.columns = df.columns
            self.cdict = df.cdict
            self.data = df.data
            assert ix is not None
        self.ix = np.arange(len(self.data[0]), dtype=np.int64) if ix is None else ix
 
    def __getitem__(self, cname):
        """
        Returns dataset column.
        :param cname: column name
        :return: the column as a numpy array
        """
        return self.data[self.cdict[cname]][self.ix]
 
    def __len__(self):
        """
        The number of samples
        :return:
        """
        return len(self.ix)
 
    def to_dict(self):
        """
        Return the data in a form used in prediction.
        :return: list of dicts with dict for each data sample, keys are the column names
        """
        return [{c: self.data[self.cdict[c]][i] for c in self.columns} for i in self.ix]
 
    def modify_col(self, cname, d):
        """
        Creates a copy of this dataset replacing one of its columns data. This method might be helpful for the Gradient
        Boosted Trees.
        :param cname: column name
        :param d: a numpy aray with new column data
        :return: new Dataset
        """
        assert len(self.ix) == len(self.data[0]), 'works for unfiltered rows, only'
        new_dataset = Dataset(self, ix=self.ix)
        new_dataset.data = list(self.data)
        new_dataset.data[self.cdict[cname]] = d
        return new_dataset
 
    def filter_rows(self, cname, cond):
        """
        Creates a new Dataset containing only the rows satisfying a given condition.
        :param cname: column name
        :param cond: condition
        :return:
        """
        col = self[cname]
        return Dataset(self, ix=self.ix[cond(col)])
 
 
def generate_sin_data(n, random_x=False, scale=0.0):
    """
    Sin dataset generator.
    """
    rng = np.random.RandomState(1234)
    if random_x:
        X = rng.uniform(0, 2 * np.pi, n)
    else:
        X = np.linspace(0, 2 * np.pi, n)
    T = np.sin(X) + rng.normal(0, scale, size=X.shape)
    df = pd.DataFrame({'x': X, 't': T}, columns=['x', 't'])
    #print("df:\n", df)
    return Dataset(df), np.sqrt(np.mean((T - np.sin(X)) ** 2))
 
 
def generate_boston_housing():
    """
    Import Boston housing.
    """
    # https://www.kaggle.com/c/boston-housing/data
    df = pd.read_csv('housing.csv')
    df.drop(['ID'], axis=1, inplace=True)  # remove unwanted column
    data_housing_train, data_housing_test = train_test_split(df, test_size=0.3, random_state=1)
    return Dataset(data_housing_train), Dataset(data_housing_test)
 
 
class DecisionNode(object):
    """
    Represents an inner decision node
    """
 
    def __init__(self, attr, value, left, right):
        """
        Constructs a node
        :param attr: splitting attribute
        :param value: splitting attribute value
        :param left: left child
        :param right: right child
        """
        self.attr = attr
        self.value = value
        self.left = left
        self.right = right
 
    def evaluate(self, x):
        """
        Evaluates the node.
        :param x: a dictionary: key = attribute (column) name, value = attribute value
        :return: reference to the corresponding child
        """
        if isinstance(self.value, str):
            if x[self.attr] == self.value:
                return self.left.evaluate(x)
            else:
                return self.right.evaluate(x)
        else:
            if x[self.attr] <= self.value:
                return self.left.evaluate(x)
            else:
                return self.right.evaluate(x)
 
    def get_nodes(self):
        """
        Return all nodes of the subtree rooted in this node.
        """
        ns = []
        q = deque([self])
        while len(q) > 0:
            n = q.popleft()
            ns.append(n)
            if isinstance(n, DecisionNode):
                q.append(n.left)
                q.append(n.right)
        return ns
 
    def __str__(self):
        """
        String representation f
        :return:
        """
        if isinstance(self.value, str):
            return '{}=="{}"'.format(self.attr, self.value)
        else:
            return '{}<={:5.2f}'.format(self.attr, self.value)
 
 
class LeafNode(object):
    def __init__(self, response):
        self.response = response
 
    def evaluate(self, x):
        return self.response
 
    def get_nodes(self):
        return [self]
 
    def __str__(self):
        return '{:5.2f}'.format(self.response)
 
 
class RegressionTree(object):
    def __init__(self, data, tattr, xattrs=None, max_depth=5,
                 max_features=lambda n: n,
                 rng=np.random.RandomState(1)):
        """
        Regression tree constructor. Constructs the model fitting supplied dataset.
        :param data: Dataset instance
        :param tattr: the name of target attribute column
        :param xattrs: list of names of the input attribute columns
        :param max_depth: limit on tree depth
        :param max_features: the number of features considered when splitting a node (all by default)
        :param rng: random number generator used for sampling features when selecting a split candidate
        """
        self.xattrs = [c for c in data.columns if c != tattr] if xattrs is None else xattrs
        self.tattr = tattr
        self.max_features = int(np.ceil(max_features(len(self.xattrs))))
        self.rng = rng
        self.root = self.build_tree(data, self.impurity(data), max_depth=max_depth)
 
    def evaluate(self, x):
        """
        Evaluates the node.
        :param x: a dictionary: key = attribute (column) name, value = attribute value
        :return: reference to the corresponding child
        """
        return self.root.evaluate(x)
 
    def impurity(self, data):
        """
        Impurity/loss for constant mean model of data. Squared loss used here.
        """
        if len(data) == 0:
            return 0.0
        t = data[self.tattr]
        return np.sum((t - t.mean()) ** 2)
 
    def build_tree(self, data, impurity, max_depth):
        if max_depth > 0:
            best_impurity = impurity
            xattrs = self.rng.choice(self.xattrs, self.max_features,
                                     replace=False)  # select attributes to be considered
            for xattr in xattrs:
                vals = np.unique(data[xattr])  # get all unique values of the attribute
                if len(vals) <= 1: continue
                for val in vals:
                    if isinstance(val, str):
                        data_l = data.filter_rows(xattr, lambda a: a == val)
                        data_r = data.filter_rows(xattr, lambda a: a != val)
                    else:
                        data_l = data.filter_rows(xattr, lambda a: a <= val)
                        data_r = data.filter_rows(xattr, lambda a: a > val)
 
                    impurity_l = self.impurity(data_l)
                    impurity_r = self.impurity(data_r)
 
                    split_impurity = impurity_l + impurity_r  # total impurity if splitting
 
                    if split_impurity < best_impurity and len(data_l) > 0 and len(data_r) > 0:
                        best_impurity, best_xattr, best_val = split_impurity, xattr, val
                        best_data_l, best_data_r = data_l, data_r
                        best_impurity_l, best_impurity_r = impurity_l, impurity_r
 
            if best_impurity < impurity:  # splitting reduces the impurity, choose best possible split
                return DecisionNode(best_xattr, best_val,
                                    self.build_tree(best_data_l, best_impurity_l, max_depth - 1),
                                    self.build_tree(best_data_r, best_impurity_r, max_depth - 1))
        return LeafNode(data[self.tattr].mean())
 
    def plot(self):
        from IPython.display import Image
        import networkx as nx
        g = nx.DiGraph()
        V = self.root.get_nodes()
        d = {}
        for i, n in enumerate(V):
            d[n] = i
            g.add_node(i, label='{}'.format(n))
        for n in V:
            if isinstance(n, DecisionNode):
                g.add_edge(d[n], d[n.left])
                g.add_edge(d[n], d[n.right])
 
        dot = nx.drawing.nx_pydot.to_pydot(g)
        return Image(dot.create_png())
 
 
def evaluate_all(model, data):
    """
    Makes predictions for all dataset samples.
    :param model: any model implementing evaluate(x) method
    :param data: Dataset instance
    :return: predictions as a numpy array
    """
    return np.r_[[model.evaluate(x) for x in data.to_dict()]]
 
 
def rmse(model, data):
    """
    Evaluates RMSE on a dataset
    :param model: any model implementing evaluate(x) method
    :param data: Dataset instance
    :return: RMSE as a float
    """
    ys = evaluate_all(model, data)
    rmse = np.sqrt(np.mean((data[model.tattr] - ys) ** 2))
    return rmse
 
 
class RandomForest(object):
    def __init__(self, data, tattr, xattrs=None,
                 n_trees=10,
                 max_depth=np.inf,
                 max_features=lambda n:n,
                 rng=np.random.RandomState(1)):
        """
        Random forest constructor. Constructs the model fitting supplied dataset.
        :param data: Dataset instance
        :param tattr: the name of target attribute column
        :param xattrs: list of names of the input attribute columns
        :param n_trees: number of trees
        :param max_depth: limit on tree depth
        :param max_features: the number of features considered when splitting a node (all by default)
        :param rng: random number generator
        """
        self.xattrs = [c for c in data.columns if c != tattr] if xattrs is None else xattrs
        self.tattr = tattr
        self.max_features = int(np.ceil(max_features(len(self.xattrs))))
        self.max_depth = max_depth
        self.rng = rng
        self.n_trees = n_trees
        #self.data = data
        self.trees = []
        for i in range(self.n_trees):
            new_data =  Dataset(data, np.random.choice(data.ix, len(data.ix), replace=True))
            new_tree = RegressionTree(data=new_data, tattr=self.tattr, xattrs=self.xattrs, max_depth=self.max_depth,
                                      rng=self.rng)
            self.trees.append(new_tree.build_tree(new_data, new_tree.impurity(new_data), max_depth=max_depth))
 
    def evaluate(self, x):
        ret = []
        for i in range(self.n_trees):
            res = self.trees[i].evaluate(x)
            #print("tattr is:", self.data[self.tattr])
            ret.append(res)
        return sum(ret)/len(ret)
 
 
class GradientBoostedTrees(object):
    def __init__(self, data, tattr, xattrs=None,
                 n_trees=10,
                 max_depth=1,
                beta=None,
                 rng=np.random.RandomState(1)):
        """
        Gradient Boosted Trees constructor. Constructs the model fitting supplied dataset.
        :param data: Dataset instance
        :param tattr: the name of target attribute column
        :param xattrs: list of names of the input attribute columns
        :param n_trees: number of trees
        :param max_depth: limit on tree depth
        :param beta: learning rate
        :param rng: random number generator
        """
        self.xattrs = [c for c in data.columns if c != tattr] if xattrs is None else xattrs
        self.tattr = tattr
        self.beta = beta
        self.rng = rng
        self.n_trees = n_trees
        self.data = data
        self.init_mean = np.mean(data[tattr])
        current_residuals = data[self.tattr]-np.mean(data[tattr])
        current_data = data.modify_col(self.tattr, current_residuals)
        self.trees = []
        all_values = []
        for i in range(self.n_trees):
            new_tree = RegressionTree(current_data, tattr, xattrs, max_depth, max_features=lambda n:n, rng= rng)
            self.trees.append(new_tree)
            current_value = evaluate_all(self.trees[-1], data)
            all_values.append(current_value)
            current_residue = data[tattr]-(self.init_mean+(self.beta*np.sum(all_values, axis=0)))
            current_data = data.modify_col(self.tattr, current_residue)
 
    def evaluate(self, x):
        predicted = self.init_mean
        for tree in self.trees:
            predicted += self.beta*tree.evaluate(x)
 
        return predicted
 
 
def generate_plot(ds_train, ds_test, tattr,
                  model_cls,
                  iterate_over,
                  iterate_values,
                  title,
                  xlabel,
                  rng,
                  iterate_labels=None,
                  bayes_rmse=None,
                  **model_params):
    """
    Generates plot of training and testing RMSE errors iterating over values of a selected parameter.
    :param ds_train: training Dataset instance
    :param ds_test: testing Dataset instance
    :param tattr: the name of target attribute column
    :param model_cls: model class, e.g., RegressionTree, RandomForest, GradientBoostedTrees
    :param iterate_over: the name of model parameter to iterate over (as string)
    :param iterate_values: a list of values to iterate over
    :param title: plot title
    :param xlabel: x axis label
    :param rng: random number generator
    :param iterate_labels: the labels corresponding to iterate_values, if not given, iterate_values are used instead, use for non float parameters
    :param bayes_rmse: plots the best achievable error (we know this for the sin dataset)
    :param model_params: other model parameters
    """
    if iterate_labels is None:
        iterate_coords = iterate_values
    else:
        assert len(iterate_labels) == len(iterate_values)
        iterate_coords = range(len(iterate_labels))
 
    train_rmses, test_rmses = [], []
    for val in iterate_values:
        st = time()
        params = dict(model_params)
        params[iterate_over] = val
        model = model_cls(ds_train, tattr=tattr, rng=rng, **params)
        train_rmses.append(rmse(model, ds_train))
        test_rmses.append(rmse(model, ds_test))
        print('{}: {} = {} finished in {:5.2f}s'.format(title, iterate_over, val, time() - st))
    best = np.argmin(test_rmses)
 
    plt.figure()
    plt.plot(iterate_coords, train_rmses, '.-', label='train')
    plt.plot(iterate_coords, test_rmses, '.-', label='test')
    if bayes_rmse is not None:
        plt.plot([0, iterate_coords[-1]], [bayes_rmse, bayes_rmse], '-.', label='$h^{*}(x)$')
    plt.plot(iterate_coords[best], test_rmses[best], 'o', label='best')
    plt.xlabel(xlabel)
    if iterate_labels is not None: plt.xticks(iterate_coords, iterate_labels)
    plt.ylabel('RMSE')
    plt.title('best test RMSE = {}'.format(test_rmses[best]))
    plt.suptitle(title)
    plt.legend()
 
def experiment_random_forest_sin(show=True):
    data_sin_train,_ = generate_sin_data(n=100, scale=0.2)
    data_sin_test, sin_test_rmse = generate_sin_data(n=1000, scale=0.2)
    rng = np.random.RandomState(1)
    generate_plot(data_sin_train, data_sin_test, tattr='t',
                  model_cls=RandomForest,
                  iterate_over='n_trees', iterate_values=[1, 2, 5, 10, 20, 50, 100, 200],
                  iterate_labels=[1, 2, 5, 10, 20, 50, 100, 200],title='Random Forest Tree for sin',
                  xlabel='tree count', bayes_rmse=sin_test_rmse, rng=rng, max_depth=math.inf)
    plt.savefig('random_forest_sin.pdf')
    if show: plt.show()
 
 
def experiment_random_forest_housing(show=True):
    data_housing_train,data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    generate_plot(data_housing_train, data_housing_test, tattr='medv',
                  model_cls=RandomForest,
                  iterate_over='n_trees', iterate_values=[1, 2, 5, 10, 20, 50, 100, 200],
                  iterate_labels=[1, 2, 5, 10, 20, 50, 100, 200],title='Random Forest Tree for Boston housing',
                  xlabel='tree count',  rng=rng, max_depth=math.inf)
    plt.savefig('random_forest_Boston_housing.pdf')
 
    if show: plt.show()
 
def experiment_random_forest_housing_splitting(show=True):
 
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
 
    generate_plot(data_housing_train, data_housing_test, tattr='medv',
                  model_cls=RandomForest,
                  iterate_over='max_features', iterate_values=[lambda n:2, lambda n:n/2, lambda n:n], iterate_labels=['att:2', 'att:half', 'att:all'],
                  n_trees=200, title='Random Forest Tree for Boston housing with varying attributes',
                  xlabel='attribute splitting count', rng=rng)
    plt.savefig('random_forest_Boston_housing_features.pdf')
    if show: plt.show()
    return
 
 
 
 
 
def experiment_gradient_boost_sin(show=True):
    data_sin_train, _ = generate_sin_data(n=100, scale=0.2)
    data_sin_test, sin_test_rmse = generate_sin_data(n=1000, scale=0.2)
    rng = np.random.RandomState(1)
    betas = [0.1, 0.2, 0.5, 1.0]
    for b in betas:
        generate_plot(data_sin_train, data_sin_test, tattr='t',
                      model_cls=GradientBoostedTrees,
                      iterate_over='n_trees', iterate_values=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                      iterate_labels=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], title='Gradient Boosted Trees for sin with beta = '+str(b),
                      xlabel='tree count',beta=b, bayes_rmse=sin_test_rmse, rng=rng)
        plt.savefig('gradient_boost_sin_beta='+str(b)+'.pdf')
    if show: plt.show()
 
def experiment_gradient_boost_housing(show=True):
    ddata_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    betas = [0.1, 0.2, 0.5, 1.0]
    for b in betas:
        generate_plot(ddata_housing_train, data_housing_test, tattr='medv',
                      model_cls=GradientBoostedTrees,
                      iterate_over='n_trees', iterate_values=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
                      iterate_labels=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000], title='Gradient Boosted Trees for housing',
                      xlabel='tree count',  beta = b, rng=rng, max_depth=1)
        plt.savefig('gradient_boost_housing_beta='+str(b)+'.pdf')
    if show: plt.show()
 
def experiment_tree_housing(show=False):
    data_housing_train, data_housing_test = generate_boston_housing()
    rng = np.random.RandomState(1)
    generate_plot(data_housing_train, data_housing_test, tattr='medv',
                  model_cls=RegressionTree,
                  iterate_over='max_depth', iterate_values=range(15),
                  title='Regression Tree (housing)',
                  xlabel='max depth',
                  rng=rng,
                  )
    plt.savefig('regression_tree_housing.pdf')
    if show: plt.show()
 
 
def experiment_tree_sin(show=True):
    data_sin_train, _ = generate_sin_data(n=100, scale=0.2)
    data_sin_test, sin_test_rmse = generate_sin_data(n=1000, scale=0.2)
    rng = np.random.RandomState(1)
 
    generate_plot(data_sin_train, data_sin_test, tattr='t',
                  model_cls=RegressionTree,
                  iterate_over='max_depth', iterate_values=range(15),
                  title='Regression Tree (sin)',
                  xlabel='max depth',
                  bayes_rmse=sin_test_rmse,
                  rng=rng
                  )
    plt.savefig('regression_tree_sin.pdf')
    if show: plt.show()
 
 
 
 
if __name__ == '__main__':
    # Assignment 1
    experiment_tree_sin(show=True)
    # experiment_tree_housing(show=True)
    # Assignment 2
    #experiment_random_forest_sin(show=True)
    #experiment_random_forest_housing(show=True)
    #experiment_random_forest_housing_splitting(show=True)
    # Assignment 3
    #experiment_gradient_boost_sin(show=True)
    #experiment_gradient_boost_housing(show=True)
