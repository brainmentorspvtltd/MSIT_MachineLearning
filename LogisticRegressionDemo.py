import math

dataset = [
    [2.78, 2.55, 0],
    [1.46, 2.36, 0],
    [3.39, 4.40, 0],
    [1.38, 1.85, 0],
    [3.06, 3.00, 0],
    [7.62, 2.75, 1],
    [5.33, 2.08, 1],
    [6.92, 1.77, 1],
    [8.67, -0.24, 1],
    [7.67, 3.50, 1]
    ]

coef = [-0.406, 0.852, -1.104]

def predict(coef, row): #coef => B
    y_pred = coef[0]
    for i in range(len(row) - 1):
        y_pred += coef[i+1] * row[i]
    return 1 / (1 + math.exp(-y_pred) )

for row in dataset:
    prediction = predict(coef, row)
    print(f"Prediction : {prediction}, Actual : {row[-1]}, Expected : {round(prediction)}")