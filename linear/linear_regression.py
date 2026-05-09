import time

start_time = time.perf_counter()
data = [[10,20,30],[2,3,4]]
target = [50,90,130]
weight = [0] * (len(data)+1)
lr = 0.0001
epoch = 100
def gradient(data,target,weight,idcol):
    n = len(data)
    row = len(data[0])
    result = 0
    if idcol == len(data):
        for i in range(row):
            y = 0
            for j in range(n):
                y += data[j][i] * weight[j]
            y += weight[n]
            y -= target[i]
            result += y 
            
        result = 2 / row * result
        return result
    else:
        for i in range(row):
            y = 0
            for j in range(n):
                y += data[j][i] * weight[j]
            y += weight[n]
            y -= target[i]
            y  *= data[idcol][i]
            result +=  y
        result = 2 / row * result
        return result
def mae(data,target,weight):
    result = 0
    n = len(data)
    row = len(data[0])
    for i in range(row):
        y = 0
        for j in range(n):
            y += data[j][i] * weight[j]
        y += weight[n]
        y  = abs(y - target[i])
        result += y 
    result = result/ row
    return result
        
n = len(data[0])
#print(weight)

for train in range(epoch):
    grad = [0] * (len(data)+1)
    for i in range(len(data)):
        grad[i] = gradient(data,target,weight,i)
        #print(gradient(data,target,weight,i))
    grad[len(data)] = gradient(data,target,weight,len(data))
    for j in range(len(data)+1):
        weight[j] -= lr * grad[j]
        #print(weight[j])
    print(f"[{train+1}] MAE : {mae(data,target,weight)}")
end_time = time.perf_counter()
print(f"Runtime: {end_time - start_time:.6f} seconds")
print(weight)
        
    
