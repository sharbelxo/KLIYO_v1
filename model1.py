import pandas as pd
import numpy as np
import math

# cleaning dataset
df = pd.read_csv(r"C:\Users\Sharbel\VSCode\Discord\datasets\StudentsPerformance.csv")
df = df[["math score","reading score"]]
df.rename(columns = {"math score" : "Exam 1", "reading score" : "Exam 2"}, inplace = True)
df["Admission"] = 0
for index, row in df.iterrows():
    if (row["Exam 1"] + row["Exam 2"]) / 2 > 60:
        row["Admission"] = 1

# convert dataset into training data x and y
exam_grades = df[["Exam 1", "Exam 2"]]
x_train = exam_grades.to_numpy()
# array([[72, 72],
#        [69, 90],
#        [90, 95],
#        ...,
#        [59, 71],
#        [68, 78],
#        [77, 86]])
y_train = df["Admission"].to_numpy()
# array([1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
#        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1,
#        0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1,
#        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1,
#        1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

# logistic regression model
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

def compute_cost(X, y, w, b, lambda_= 1):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost /= m
    return cost

def compute_gradient(X, y, w, b, lambda_=None): 
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb = sigmoid(z_i)
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        for j in range(n):
            dj_dw_ij = (f_wb - y[i])* X[i][j]
            dj_dw[j] += dj_dw_ij      
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    J_history = []
    w_history = []
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
        if i<100000:
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
    return w_in, b_in, J_history, w_history

def predict(X, w, b):
    m = X.shape[0]
    p = np.zeros(m)
    for i in range(m):   
        z_i = np.dot(X[i], w) + b
        f_wb = 1 / (1 + np.exp(-z_i))
        p[i] = f_wb >= 0.5
    return p

def run_gradient_descent():
    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2).reshape(-1,1) - 0.5)
    initial_b = -8
    iterations = 10000
    alpha = 0.001
    w,b,_,_ = gradient_descent(x_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)
    return w,b

w,b = run_gradient_descent()

p = predict(x_train, w,b)
accuracy = str(np.mean(p == y_train) * 100) + "%"
print(accuracy)

w_str = str(w).replace(' [', '').replace('[', '').replace(']', '')
b_str = str(b).replace(' [', '').replace('[', '').replace(']', '')

file = open("wb1.txt","w")
file.write(str(w_str) + "\n" + str(b_str))
file.close()