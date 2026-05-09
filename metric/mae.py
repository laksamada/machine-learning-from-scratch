def mae(y_pred,target):
    result = 0
    m = len(target)
    for i in range(m):
        result += abs(y_pred[i]-target[i])
    return result/m