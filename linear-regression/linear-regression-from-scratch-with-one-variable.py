import pandas as pd
import matplotlib.pyplot as plt

filePath = "/home/filfilu/python/ML/dataset/studytime_score_100.csv"
dataset = pd.read_csv(filePath)

def Model(w,b,study_time):
    return w * study_time + b
def gradient_descent(w_now,b_now,points,L): # L stands for learning rate
    w_gradient = 0
    b_gradient = 0

    n = len(points)
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        w_gradient += ((w_now * x + b_now) - y)*x
        b_gradient += ((w_now * x + b_now) - y)
    w = w_now - (L/n) * w_gradient
    b = b_now - (L/n) * b_gradient
    return w,b
L = 0.01
w,b = 0,0
epoch = 1000
for i in range(epoch):
    w,b = gradient_descent(w,b,dataset,L)


plt.scatter(dataset.studytime,dataset.score,color="blue")
plt.plot(list(range(1,11)),[w *x + b for x in range(1,11)],color="red")
plt.show()




