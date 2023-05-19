import openml
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay

class SVM():
    def __init__(self, C, kernel, epsilon):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.bias = 0

    def kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'gaussian':
            if np.ndim(x1) == 1 and np.ndim(x2) == 1:
                result = np.exp(- (np.linalg.norm(x1 - x2, 2)) ** 2 / (2 * 1 ** 2))
            elif (np.ndim(x1) > 1 and np.ndim(x2) == 1) or (np.ndim(x1) == 1 and np.ndim(x2) > 1):
                result = np.exp(- (np.linalg.norm(x1 - x2, 2, axis=1) ** 2) / (2 * 1 ** 2))
            elif np.ndim(x1) > 1 and np.ndim(x2) > 1:
                result = np.exp(- (np.linalg.norm(x1[:, np.newaxis] - x2[np.newaxis, :], 2, axis=2) ** 2) / (2 * 1 ** 2))
            return result
        elif self.kernel == 'polynomial':
            return np.dot(x1, x2)**3
        elif self.kernel == 'laplacian':
            return np.exp(-np.linalg.norm(x1-x2) / 2)
        elif self.kernel == 'exponential':
            return np.exp(-5*np.linalg.norm(x1-x2))
      
    def fit(self, train_x, train_y, max_iter=1000):
        self.train_x = np.array(train_x)
        self.train_y = np.array(train_y)
        self.alpha = np.zeros(train_x.shape[0])
        iter = 0
        converged = False
        while iter < max_iter and not converged:
            print('iter = {}'.format(iter))
            index1, index2 = self.SMO_get_alpha()
            if index1 == -1:
                converged = True
                print('End of iteration, iter = {}'.format(iter))
                break
            converged = self.SMO_train(index1, index2)
            iter += 1
            if converged:
                print('End of iteration, iter = {}'.format(iter))
                break

    def SMO_get_alpha(self):
        for i in range(self.alpha.shape[0]): # iterate through range of alpha values
            satisfied_conditions = False
            if 0 < self.alpha[i] < self.C and self.train_y[i] * self.decision_function(self.train_x[i]) != 1: # checks if satisfies 0 < self.alpha[i] < self.C, and check if violates KKT
                satisfied_conditions = True # set to True
            elif self.alpha[i] == 0 and self.train_y[i] * self.decision_function(self.train_x[i]) < 1: # if not true, check this
                satisfied_conditions = True
            elif self.alpha[i] == self.C and self.train_y[i] * self.decision_function(self.train_x[i]) > 1:
                satisfied_conditions = True
            if satisfied_conditions: # if true
                index2 = self.choose_another_alpha(i) # calls method to select another alpha value index 2 that will be optimized together with i
                return i, index2
        return -1, -1 # return -1, -1 to indicate that no suitable alpha values were found

    def decision_function(self, x):
        return np.sum(self.alpha * self.train_y * self.kernel_function(self.train_x, x)) + self.bias

    def error(self, index):
        y_pred = self.decision_function(self.train_x[index])
        y_true = self.train_y[index]
        return y_pred - y_true

    def choose_another_alpha(self, index):
        errors = [np.abs(self.error(index) - self.error(i)) for i in range(self.alpha.shape[0])] # iterates through the range of alpha values in the training set, compute a list of errors 
        errors[index] = 0 # excluding the current alpha value at "index" by setting it 0
        result_index = np.argmax(errors) # get the index of the alpha value that has the largest error difference with the current alpha value
        return result_index
  
    def SMO_train(self, index1, index2):
        old_alpha = self.alpha.copy()
        x1 = self.train_x[index1]
        y1 = self.train_y[index1]
        x2 = self.train_x[index2]
        y2 = self.train_y[index2]

        alpha2 = self.compute_alpha2(index1, index2, x1, x2, y1, y2, old_alpha)
        L, H = self.compute_LH(y1, y2, old_alpha, index1, index2)
        alpha2 = self.clip_alpha(alpha2, L, H)
        alpha1 = old_alpha[index1] + y1 * y2 * (old_alpha[index2] - alpha2)

        self.alpha[index1] = alpha1
        self.alpha[index2] = alpha2

        b1 = self.compute_b(x1, x2, y1, y2, alpha1, alpha2, old_alpha, index1, index2)
        b2 = self.compute_b(x1, x2, y1, y2, alpha1, alpha2, old_alpha, index2, index1)

        self.bias = self.compute_bias(alpha1, alpha2, y1, y2, b1, b2)

        e = np.linalg.norm(old_alpha - self.alpha)
        print('E = {}'.format(e))

        if e < self.epsilon:
            return True
        else:
            return False

    def compute_alpha2(self, index1, index2, x1, x2, y1, y2, old_alpha):
        eta = self.kernel_function(x1, x1) + self.kernel_function(x2, x2) - 2 * self.kernel_function(x1, x2)
        return old_alpha[index2] + y2 * (self.error(index1) - self.error(index2)) / eta

    def compute_LH(self, y1, y2, old_alpha, index1, index2):
        if y1 != y2:
            L = max(0, old_alpha[index2] - old_alpha[index1])
            H = min(self.C, self.C + old_alpha[index2] - old_alpha[index1])
        else:
            L = max(0, old_alpha[index1] + old_alpha[index2] - self.C)
            H = min(self.C, old_alpha[index1] + old_alpha[index2])
        return L, H

    def clip_alpha(self, alpha2, L, H):
        if alpha2 > H:
            alpha2 = H
        elif alpha2 < L:
            alpha2 = L
        return alpha2

    def compute_b(self, x1, x2, y1, y2, alpha1, alpha2, old_alpha, index, index_other):
        return -self.error(index) - y1 * self.kernel_function(x1, x1) * (alpha1 - old_alpha[index]) - y2 * self.kernel_function(x1, x2) * (alpha2 - old_alpha[index_other]) + self.bias

    def compute_bias(self, alpha1, alpha2, y1, y2, b1, b2):
        if 0 < alpha1 < self.C:
            return b1
        elif 0 < alpha2 < self.C:
            return b2
        else:
            return (b1 + b2) / 2

    def transform_one(self, x):
        if self.decision_function(x) > 0:
            return 1
        else:
            return -1

    def predict(self, test_x):
        return np.array([self.transform_one(x) for x in test_x])



if __name__ == '__main__':
    start_time = time.time()
    # load dataset
    dataset = openml.datasets.get_dataset(40945)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="dataframe", target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    y=y.astype('float')
    df["survived"] = y
    # feature engineering
    title = df.name.str.split(".").str.get(0).str.split(",").str.get(-1)
    title.replace(to_replace = ["Dr", "Rev", "Col", "Major", "Capt"], value = "Officer", inplace = True,regex=True)
    title.replace(to_replace = ["Dona", "Jonkheer", "Countess", "Sir", "Lady", "Don"], value = "Aristocrat", inplace = True,regex=True)
    title.replace({"Mlle":"Miss", "Ms":"Miss", "Mme":"Mrs"}, inplace = True,regex=True)
    title.replace({"the Aristocrat":"Aristocrat"}, inplace = True,regex=True)
    df["pName"] = title
    df["pFamily"] = df.sibsp + df.parch + 1
    df.pFamily.replace(to_replace = [1], value = "single", inplace = True)
    df.pFamily.replace(to_replace = [2,3], value = "small", inplace = True)
    df.pFamily.replace(to_replace = [4,5], value = "medium", inplace = True)
    df.pFamily.replace(to_replace = [6, 7, 8, 11], value = "large", inplace = True)
    def calculateMissingValues(variable):
        return df.isna().sum()[df.isna().sum()>0]
    df.embarked.fillna(value="S", inplace = True)
    df.fare.fillna(value=df.fare.median(), inplace = True)
    df.age = df.groupby(["pName", "pclass"])["age"].transform(lambda x: x.fillna(x.median()))
    ageGroups = ["infant","child","adolescents ","adults","oldAdult","elderly"]
    groupRanges = [0,1,12,17,35,65,81]
    df["ageBinned"] = pd.cut(df.age, groupRanges, labels = ageGroups)
    fareGroups = ["low","medium","high","veryHigh"]
    fareGroupRanges = [-1, 130, 260, 390, 520]
    df["fareBinned"] = pd.cut(df.fare, fareGroupRanges, labels = fareGroups)
    df = df[["pclass", "sex", "embarked", "pName", "pFamily", "ageBinned", "fareBinned", "survived"]]
    df.loc[:, ["pclass","pName", "pFamily", "survived"]]= df.loc[:, ["pclass","pName", "pFamily", "survived"]].astype('category')
    df.loc[:, ["survived"]]= df.loc[:, ["survived"]].astype('float')
    df = pd.get_dummies(df)
    df.survived=df.survived.replace(0, -1)
    X = df.drop('survived', axis=1)
    y = df['survived'].astype('float').to_numpy()

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 20)

    svm = SVM(1, 'linear', 0.001)
    svm.fit(X_train, y_train)
    predict_label = svm.predict(X_test)

    accuracy = accuracy_score(y_test, predict_label)
    precision = precision_score(y_test, predict_label)
    recall = recall_score(y_test, predict_label)
    tn, fp, fn, tp = confusion_matrix(y_test, predict_label).ravel()
    specificity = tn / (tn + fp)
    print("The accuracy score is: ", accuracy)
    print("The precision score is: ", precision)
    print("The recall score is: ", recall)
    print("The specificity score is: ", specificity)
    cm = confusion_matrix(y_test, predict_label)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    '''
    # plot hyperplane, commented out because for some kernel functions/hyperparameter settings, hyperplane cannot be plotted
    points = []
    for i in np.linspace(-4, 4, num=400):
      for j in np.linspace(-4, 4, num=400):
        x_ij = np.array([i, j])
        if -0.01 < svm.decision_function(x_ij) < 0.01:
          tmp = [i, j]
          points.append(tmp)

    points = np.array(points)
    print(points)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
    plt.scatter(points[:, 0], points[:, 1], marker='o')

    plt.show()
    '''
    print(f"Execution time: {time.time() - start_time}")
