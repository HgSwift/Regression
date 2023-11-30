import scipy.io
mat = scipy.io.loadmat('test_data.mat')

print(mat['X'])