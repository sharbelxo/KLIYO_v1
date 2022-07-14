import numpy as np
import pandas as pd
import copy, math

# preparing and cleaning data for the model to use

data = pd.read_csv(r"C:\Users\Sharbel\VSCode\Discord\datasets\insurance.csv")
rows = data.shape[0]
columns = data.shape[1]
max_age = float(data["age"].max())
max_bmi = float(data["bmi"].max())
max_children = float(data["children"].max())
max_charges = float(data["charges"].max())
smokers = data[(data.smoker == "yes")]
non_smokers = data[(data.smoker == "no")]
data = data.dropna()
data.drop(["region", "sex"], axis = 1, inplace = True)
smoking = {"no": 0, "yes": 1}
data["smoker"] = data["smoker"].apply(lambda x: smoking[x])

# normalizing by max
normalize_data0 = data.divide(data.max())

# normalizing by z-score
normalize_data1 = (data - data.mean()).divide(data.std())

# normalizing by mean
normalize_data2 = (data - data.mean()).divide(data.max() - data.min())

# we are going to choose normalizing by max (easier to use later on when we get user input)
data = normalize_data0

# multivariate linear regression model

x_train = data[["age", "bmi", "children", "smoker"]]
y_train = data["charges"]

x_train = x_train.to_numpy()
y_train = y_train.to_numpy()

w_initial = np.zeros(shape = 4)
b_initial = 0.

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2
    cost = cost / (2 * m)
    return cost

def compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        err = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db,dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i<100000:
            J_history.append( cost_function(X, y, w, b))
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history

def run_gradient_descent():
    # I think having initial_w and initial_b that are better than a zero would make the model better
    initial_w = np.zeros(shape = 4)
    initial_b = 0.
    iterations = 1000
    alpha = 0.001
    w_final, b_final,_ = gradient_descent(x_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m,_ = x_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(x_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    return w_final, b_final

w, b = run_gradient_descent()

w_str = str(w).replace(' [', '').replace('[', '').replace(']', '')

file = open("wb2.txt","w")
file.write(str(w_str) + "\n" + str(b))
file.close()
