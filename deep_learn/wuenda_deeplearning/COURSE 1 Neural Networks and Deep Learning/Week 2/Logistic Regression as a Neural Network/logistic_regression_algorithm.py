import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x, train_set_y, test_set_x, test_set_y, classes = load_dataset()
plt.imshow(train_set_x[25])

# 向量化
train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T
test_set_x = train_set_x.reshape(test_set_x.shape[0], -1).T
# 标准化
train_set_x = train_set_x/255
test_set_x = test_set_x/255


# 激活函数sigmoid
def sigmoid(x):
    return 1.0/(1.0+np.exp(-1.0*x))


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert w.shape == (dim, 1)
    assert isinstance(b, float) or isinstance(b, int)
    return w, b

def optimize():
    pass
def predict():
    pass


def mod():
    pass

