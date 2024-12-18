import math
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('../DataSets/Reliance.csv')
x = data.loc[:, 'Open']
actual_y = data.loc[:, 'Close']

print(x)
print(actual_y)
def loss_function(m, c):
    current_error = 0
    for i in range(len(data)):
        x_item = data.iloc[i].Open
        y_item = data.iloc[i].Close
        current_error += (y_item - (m * x_item + c)) ** 2
    current_error = math.sqrt(current_error / len(x))
    return current_error

def gradient_descent(m_now, c_now, L):
    m_gradient = 0
    c_gradient = 0
    n = len(data)

    for i in range(len(data)):
        x = data.iloc[i].Open
        y = data.iloc[i].Close

        m_gradient += -(2 / n) * x * (y - (m_now * x + c_now))
        c_gradient += -(2 / n) * (y - (m_now * x + c_now))

    m = m_now - m_gradient * L
    c = c_now - c_gradient * L
    return m, c

m = 1
c = 0
L = 0.0000000001
epochs = 100
final_error = 999999

for i in range(epochs):
    if i % 50 == 0:
        print(f" Epoch : {i}")
    cal_m, cal_c = gradient_descent(m, c, L)
    cal_error = loss_function(m, c)
    if cal_error < final_error:
        m = cal_m
        c = cal_c
        final_error = cal_error
        print(f" Epoch : {i} >> error : {final_error}, m : {m}, c : {c}")
    print(f" Epoch : {i} >> error : {cal_error}")

print(m, c)
plt.plot(data.Open, data.Close)

plt.plot(list(range(200, 3000)), [m * x + c for x in range(200, 3000)])

plt.show()
