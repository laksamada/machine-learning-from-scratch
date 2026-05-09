import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from metric.mae import mae

start_time = time.perf_counter()

data = [[10, 20, 30], [2, 3, 4]]
target = [50, 90, 130]

weight = [0] * (len(data) + 1)

lr = 0.0001
epoch = 100


def predict(data, weight):
    feature_count = len(data)
    row_count = len(data[0])

    preds = [0] * row_count

    for i in range(row_count):
        for j in range(feature_count):
            preds[i] += data[j][i] * weight[j]

        preds[i] += weight[-1]

    return preds


def compute_error(y_pred, target):
    err = y_pred.copy()

    for i in range(len(err)):
        err[i] -= target[i]

    return err


def compute_gradient(data, err):
    feature_count = len(data)
    row_count = len(err)

    grad = [0] * (feature_count + 1)

    for i in range(feature_count):
        total = 0

        for j in range(row_count):
            total += err[j] * data[i][j]

        grad[i] = (2 / row_count) * total

    grad[-1] = (2 / row_count) * sum(err)

    return grad


def update_weight(weight, grad, lr):
    for i in range(len(weight)):
        weight[i] -= lr * grad[i]


for train in range(epoch):
    y_pred = predict(data, weight)

    err = compute_error(y_pred, target)

    grad = compute_gradient(data, err)

    update_weight(weight, grad, lr)

    y_pred = predict(data, weight)

    print(f"[{train + 1}] MAE : {mae(y_pred, target)}")

end_time = time.perf_counter()

print(weight)
print(f"Runtime: {end_time - start_time:.6f} seconds")