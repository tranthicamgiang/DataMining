import numpy
import NeuralNetwork
from datetime import date
import PSO


def scaleData(X, a=0, b=0):
    if a == 0 and b == 0:
        max_x = X.max()
        min_x = X.min()
        high = 0.999
        low = 0.111
        a = (high - low) / (max_x - min_x)
        b = (max_x * low - min_x * high) / (max_x - min_x)
    return a * X + b, a, b


def measure_date_in_a_year(data):
    result = []
    day_pre = data[0]
    i = 0
    for next_day in data:
        if next_day.year == day_pre.year:
            i = i + 1
        else:
            i = 0
        result.append(i)
        day_pre = next_day
    return result


def readWeight(s):
    f = open(s, 'r')
    f_line = f.readlines()
    f.close()

    f_fm = [float(row.replace("\n", "")) for row in f_line]
    return numpy.asarray(f_fm)


def readData(f):
    data_file = open(f, 'r')
    data_list = data_file.readlines()
    data_file.close()
    return data_list


def size_weights(shape):
    x = 0
    for i in range(len(shape) - 1):
        x = x + (shape[i] + 1) * shape[i + 1]
    return x


def vector_to_weights(vector, shape):
    weight = []
    index = 0
    for i in range(len(shape) - 1):
        row = shape[i + 1]
        col = shape[i] + 1
        id_min = index
        id_max = index + row * col
        weight.append(vector[id_min:id_max].reshape(row, col))
    return weight


def test(nn, weight, a_x, b_x, a_y, b_y, do_am):
    # x_test = int(input('Nhap gia tri do am: '))
    # while x_test != -1:
    #     inp, a_inp, b_inp = scaleData(numpy.asarray([x_test]), a_x, b_x)
    #     print('Luong mua (', x_test, ') la : ',
    #           (nn.query(inp, vector_to_weights(weight, nn.shape)) - b_y) / a_y)
    #     x_test = int(input('Nhap gia tri do am: '))
    y_predict = nn.query(do_am, vector_to_weights(weight, nn.shape))
    # y_predict = (y_predict - b_y)/a_y
    return y_predict


def test2(nn, weight, a_doam, b_doam, a_date, b_date, a_y, b_y):
    while 1 == 1:
        inp1 = int(input("Nhap Ngay Thu: "))
        inp2 = int(input("Nhap do am"))
        inp1 = a_date * inp1 + b_date
        inp2 = a_doam * inp2 + b_doam
        X = numpy.asarray([inp1]).reshape(1, 1)
        X = numpy.c_[X, numpy.asarray([inp2])]
        y_predict = nn.query(X, vector_to_weights(weight, nn.shape))
        print("luong mua du doan: ", (y_predict - b_y) / a_y)


def Accuracy(y, y_hat):
    return 100 - (100 * (numpy.sum(abs(y - y_hat) / y)) / y.shape[0])


# read data .........
data = [row.split(",") for row in readData("data.csv")]

do_am = numpy.asarray([float(row[1]) for row in data])
do_am, a_doam, b_doam = scaleData(do_am)
do_am = do_am.reshape(len(data), 1)
luong_mua, a_y, b_y = scaleData(numpy.asarray([float(row[2].replace("\n", "")) for row in data]))
luong_mua = luong_mua.reshape(len(data), 1)
ngay_thang_nam_split = [row[0].split("/") for row in data]
ngay_thang_nam = [date(int(row[2]), int(row[1]), int(row[0])) for row in ngay_thang_nam_split]
dates = numpy.asarray(measure_date_in_a_year(ngay_thang_nam))
dates, a_date, b_date = scaleData(dates)

X = dates.copy().reshape(dates.shape[0], 1)
X = numpy.c_[X, do_am]

# divide into train and test data
X_train = X[0:15485]
X_test = X[15485:]
y_train = luong_mua[0:15485]
y_test = luong_mua[15485:]

# train .............
shape = (X.shape[1], 50, 50, 1)

# pso to find best weight for neural network
swarm = PSO.PSO(100, size_weights(shape), y_train, X_train, shape)
best_weight, best_cost = swarm.update()
print('best weight, best cost: ', best_weight, "+++++++",  best_cost)

f = open("weight.txt", "w")
for w in best_weight:
    f.write(str(w) + "\n")
f.close()
# ////////////////////////////////////////////
best_weight = readWeight("weight.txt")
best_nn = NeuralNetwork.NeuralNetwork(shape)
y_hat = test(best_nn, best_weight, a_date, b_date, a_y,  b_y, X_test)

print(Accuracy(y_test, y_hat))
xxx = (y_hat - b_y)/a_y
# test2(best_nn, best_weight, a_doam, b_doam, a_date, b_date, a_y, b_y)

