import numpy as np

if __name__ == '__main__':
    lst = np.array([[1, 2, 4], [5, 6, 7]])
    print(np.argmax(lst, 1))
