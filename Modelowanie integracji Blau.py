import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dt = 0.01
t = 10000   # t = 3000/dt => t = 3000/0.00001
alpha = 0.95
v_const = 1/(40 * 40/2)
p = 0
percent = 0
all_results = {}
list_of_keys = []
list_of_values = []
list_of_x = []
list_of_y = []

v = np.zeros((40, 60))
second_v = np.zeros((40, 60))
w = np.zeros(4)
print(w)


def assess_array(array, some_const):
    for x in range(40):
        for y in range(20):
            array[x][y] = some_const


def assess_w_1(matrix_1, first_x, volume):
    matrix_1[0] = 0
    for second_x in range(1, first_x-1):
        matrix_1[0] = matrix_1[0] + volume[second_x][y]
    matrix_1[0] = alpha * matrix_1[0]


def assess_w_2(matrix_1, volume):
    matrix_1[1] = 0
    for second_x in range(x+1, 41):
        matrix_1[1] = matrix_1[1] + volume[second_x][y]
    matrix_1[1] = (1-alpha) * matrix_1[1]


def assess_w_3(matrix_1, first_x, volume):
    matrix_1[2] = 0
    for second_x in range(1, first_x):
        matrix_1[2] = matrix_1[2] + volume[second_x][y]
    matrix_1[2] = alpha * matrix_1[2]


def assess_w_4(matrix_1, volume):
    matrix_1[3] = 0
    for second_x in range(x+1, 41):
        matrix_1[3] = matrix_1[3] + volume[second_x][y-1]
    matrix_1[3] = (1-alpha) * matrix_1[3]


assess_array(v, v_const)

for i in range(t):
    for x in range(1, 41):
        for y in range(1, 61):

            w[0] = 0
            for x2 in range(1, x):
                w[0] = w[0] + v[x2][y-1]
            w[0] = alpha * w[0]

            w[1] = 0
            for x2 in range(1, x):
                w[1] = w[1] + v[x2][y-1]
            w[1] = (1-alpha) * w[1]

            w[2] = 0
            for x2 in range(1, x):
                w[2] = w[2] + v[x2][y-1]
            w[2] = alpha * w[2]

            w[3] = 0
            for x2 in range(1, x):
                w[3] = w[3] + v[x2][y-2]
            w[3] = (1-alpha) * w[3]

            second_v[x-1][y-1] = v[x-1][y-1]+dt*(-v[x-1][y-1] * (w[0]+w[1])+v[x-1][y-1]*w[2]+v[x-1][y-2]*w[3])

    for x in range(1, 41):
        for y in range(1, 61):
            v[x-1][y-1] = second_v[x-1][y-1]

    p = p+1
    percent = (p * 100)/t
    print("We're at: ", percent, "%")


for x in range(40):
    for y in range(60):
        all_results[x, y] = v[x][y]

for key, value in all_results.items():
    list_of_keys.append(key)
    list_of_values.append(value)

for i in range(len(list_of_keys)):
    list_of_x.append(list_of_keys[i][0])
    list_of_y.append(list_of_keys[i][1])

data = [[list_of_x], [list_of_y], [list_of_values]]
df_1 = pd.DataFrame(list(zip(list_of_x, list_of_y, list_of_values)), columns=['x', 'y', 'values'])
df_1.plot.scatter(x='x', y='y', c='values', colormap='jet')
plt.show()