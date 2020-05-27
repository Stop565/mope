import math
import random
from functools import partial

import numpy as np
from scipy.stats import f, t
from prettytable import PrettyTable

def get_fisher_critical(prob,f3, f4):
    for i in [j*0.001 for j in range(int(10/0.001))]:
        if abs(f.cdf(i,f4,f3)-prob) < 0.0001:
            return i


def get_student_critical(prob, f3):
    for i in [j*0.0001 for j in range(int(5/0.0001))]:
        if abs(t.cdf(i,f3)-(0.5 + prob/0.1*0.05)) < 0.000005:
            return i

def get_cohren_critical(prob, f1, f2):
    f_crit = f.isf((1-prob)/f2, f1, (f2-1)*f1)
    return f_crit/(f_crit+f2-1)

m = 3
N = 8

p = .95
q = 1 - p

x1min = -6
x1max = 9
x2min = -9
x2max = 3
x3min = -6
x3max = 9

X_max = [x1max, x2max, x3max]
X_min = [x1min, x2min, x3min]

x_av_min = (x1min + x2min + x3min) / 3
x_av_max = (x1max + x2max + x3max) / 3

Y_max = int(round(200 + x_av_max, 0))
Y_min = int(round(200 + x_av_min, 0))
X0 = 1


X_matr = [
    [-1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1]]


x_for_beta = [
    [1, -1, -1, -1],
    [1, -1, -1, 1],
    [1, -1, 1, -1],
    [1, -1, 1, 1],
    [1, 1, -1, -1],
    [1, 1, -1, 1],
    [1, 1, 1, -1],
    [1, 1, 1, 1]
]
x_12_13_23 = [
    [1, 1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, 1],
    [-1, 1, -1],
    [1, -1, -1],
    [1, 1, 1],
]
x_123 = [
    -1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    1
]
X_matr_natur = [
    [x1min, x2min, x3min],
    [x1min, x2min, x3max],
    [x1min, x2max, x3min],
    [x1min, x2max, x3max],
    [x1max, x2min, x3min],
    [x1max, x2min, x3max],
    [x1max, x2max, x3min],
    [x1max, x2max, x3max],
]
x_12_13_23_natur = [[X_matr_natur[j][0] * X_matr_natur[j][1], X_matr_natur[j][0] * X_matr_natur[j][2],
                     X_matr_natur[j][1] * X_matr_natur[j][2]] for j in range(N)]
x_123_natur = [X_matr_natur[j][0] * X_matr_natur[j][1] * X_matr_natur[j][2] for j in range(N)]

Y_matr = [[random.randint((Y_min), (Y_max)) for i in range(m)] for j in range(N)]

Y_average = [sum(j) / m for j in Y_matr]

results_nat = [
        sum(Y_average),
        sum([Y_average[j] * X_matr_natur[j][0] for j in range(N)]),
        sum([Y_average[j] * X_matr_natur[j][1] for j in range(N)]),
        sum([Y_average[j] * X_matr_natur[j][2] for j in range(N)]),
        sum([Y_average[j] * x_12_13_23_natur[j][0] for j in range(N)]),
        sum([Y_average[j] * x_12_13_23_natur[j][1] for j in range(N)]),
        sum([Y_average[j] * x_12_13_23_natur[j][2] for j in range(N)]),
        sum([Y_average[j] * x_123_natur[j] for j in range(N)]),
    ]

mj0 = [N,
           sum([X_matr_natur[j][0] for j in range(N)]),
           sum([X_matr_natur[j][1] for j in range(N)]),
           sum([X_matr_natur[j][2] for j in range(N)]),
           sum([x_12_13_23_natur[j][0] for j in range(N)]),
           sum([x_12_13_23_natur[j][1] for j in range(N)]),
           sum([x_12_13_23_natur[j][2] for j in range(N)]),
           sum([x_123_natur[j] for j in range(N)]),
           ]
mj1 = [sum([X_matr_natur[j][0] for j in range(N)]),
           sum([X_matr_natur[j][0] ** 2 for j in range(N)]),
           sum([x_12_13_23_natur[j][0] for j in range(N)]),
           sum([x_12_13_23_natur[j][1] for j in range(N)]),
           sum([(X_matr_natur[j][0] ** 2) * X_matr_natur[j][1] for j in range(N)]),
           sum([(X_matr_natur[j][0] ** 2) * X_matr_natur[j][2] for j in range(N)]),
           sum([x_123_natur[j] for j in range(N)]),
           sum([(X_matr_natur[j][0] ** 2) * x_12_13_23_natur[j][2] for j in range(N)]),
           ]
mj2 = [sum([X_matr_natur[j][1] for j in range(N)]),
           sum([x_12_13_23_natur[j][0] for j in range(N)]),
           sum([X_matr_natur[j][1] ** 2 for j in range(N)]),
           sum([x_12_13_23_natur[j][2] for j in range(N)]),
           sum([(X_matr_natur[j][1] ** 2) * X_matr_natur[j][0] for j in range(N)]),
           sum([x_123_natur[j] for j in range(N)]),
           sum([(X_matr_natur[j][1] ** 2) * X_matr_natur[j][2] for j in range(N)]),
           sum([(X_matr_natur[j][1] ** 2) * x_12_13_23_natur[j][1] for j in range(N)]),
           ]
mj3 = [sum([X_matr_natur[j][2] for j in range(N)]),
           sum([x_12_13_23_natur[j][1] for j in range(N)]),
           sum([x_12_13_23_natur[j][2] for j in range(N)]),
           sum([X_matr_natur[j][2] ** 2 for j in range(N)]),
           sum([x_123_natur[j] for j in range(N)]),
           sum([(X_matr_natur[j][2] ** 2) * X_matr_natur[j][0] for j in range(N)]),
           sum([(X_matr_natur[j][2] ** 2) * X_matr_natur[j][1] for j in range(N)]),
           sum([(X_matr_natur[j][2] ** 2) * x_12_13_23_natur[j][0] for j in range(N)]),
           ]
mj4 = [sum([x_12_13_23_natur[j][0] for j in range(N)]),
           sum([(X_matr_natur[j][0] ** 2) * X_matr_natur[j][1] for j in range(N)]),
           sum([(X_matr_natur[j][1] ** 2) * X_matr_natur[j][0] for j in range(N)]),
           sum([x_123_natur[j] for j in range(N)]),
           sum([x_12_13_23_natur[j][0] ** 2 for j in range(N)]),
           sum([(X_matr_natur[j][0] ** 2) * x_12_13_23_natur[j][2] for j in range(N)]),
           sum([(X_matr_natur[j][1] ** 2) * x_12_13_23_natur[j][1] for j in range(N)]),
           sum([(x_12_13_23_natur[j][0] ** 2) * X_matr_natur[j][2] for j in range(N)]),
           ]
mj5 = [sum([x_12_13_23_natur[j][1] for j in range(N)]),
           sum([(X_matr_natur[j][0] ** 2) * X_matr_natur[j][2] for j in range(N)]),
           sum([x_123_natur[j] for j in range(N)]),
           sum([(X_matr_natur[j][2] ** 2) * X_matr_natur[j][0] for j in range(N)]),
           sum([(X_matr_natur[j][0] ** 2) * x_12_13_23_natur[j][2] for j in range(N)]),
           sum([x_12_13_23_natur[j][1] ** 2 for j in range(N)]),
           sum([(X_matr_natur[j][2] ** 2) * x_12_13_23_natur[j][0] for j in range(N)]),
           sum([(x_12_13_23_natur[j][1] ** 2) * X_matr_natur[j][1] for j in range(N)]),
           ]
mj6 = [sum([x_12_13_23_natur[j][2] for j in range(N)]),
           sum([x_123_natur[j] for j in range(N)]),
           sum([(X_matr_natur[j][1] ** 2) * X_matr_natur[j][2] for j in range(N)]),
           sum([(X_matr_natur[j][2] ** 2) * X_matr_natur[j][1] for j in range(N)]),
           sum([(X_matr_natur[j][1] ** 2) * x_12_13_23_natur[j][1] for j in range(N)]),
           sum([(X_matr_natur[j][2] ** 2) * x_12_13_23_natur[j][0] for j in range(N)]),
           sum([(X_matr_natur[j][2] ** 2) * X_matr_natur[j][1] for j in range(N)]),
           sum([(x_12_13_23_natur[j][2] ** 2) * X_matr_natur[j][0] for j in range(N)]),
           ]
mj7 = [sum([x_123_natur[j] for j in range(N)]),
           sum([(X_matr_natur[j][0] ** 2) * x_12_13_23_natur[j][2] for j in range(N)]),
           sum([(X_matr_natur[j][1] ** 2) * x_12_13_23_natur[j][1] for j in range(N)]),
           sum([(X_matr_natur[j][2] ** 2) * x_12_13_23_natur[j][0] for j in range(N)]),
           sum([(x_12_13_23_natur[j][0] ** 2) * X_matr_natur[j][2] for j in range(N)]),
           sum([(x_12_13_23_natur[j][1] ** 2) * X_matr_natur[j][1] for j in range(N)]),
           sum([(x_12_13_23_natur[j][2] ** 2) * X_matr_natur[j][0] for j in range(N)]),
           sum([x_123_natur[j] ** 2 for j in range(N)])
           ]

B_nat1 = np.linalg.solve([mj0, mj1, mj2, mj3, mj4, mj5, mj6, mj7], results_nat)  # list of B's
B_nat = list(B_nat1)

B_norm = [
        sum(Y_average) / N,
        sum([Y_average[j] * X_matr[j][0] for j in range(N)]) / N,
        sum([Y_average[j] * X_matr[j][1] for j in range(N)]) / N,
        sum([Y_average[j] * X_matr[j][2] for j in range(N)]) / N,
        sum([Y_average[j] * x_12_13_23[j][0] for j in range(N)]) / N,
        sum([Y_average[j] * x_12_13_23[j][1] for j in range(N)]) / N,
        sum([Y_average[j] * x_12_13_23[j][2] for j in range(N)]) / N,
        sum([Y_average[j] * x_123[j] for j in range(N)]) / N,
    ]

print("Матриця планування експерименту:")

tb = PrettyTable()
tb.field_names = ["N", "x1", " x2", "x3", "Y1", "Y2", "Y3"]

for i in range(N):
    tb.add_row([i + 1, X_matr[i][0], X_matr[i][1], X_matr[i][2], Y_matr[i][0], Y_matr[i][1], Y_matr[i][2]])

print(tb)

def criterion_of_Student(value, criterion, check):
    if check < criterion:
        return 0
    else:
        return value


y1_nat = B_nat[0] + B_nat[1] * X_matr_natur[0][0] + B_nat[2] * X_matr_natur[0][1] + B_nat[3] * X_matr_natur[0][2] + \
             B_nat[4] * x_12_13_23_natur[0][0] + B_nat[5] * x_12_13_23_natur[0][1] + B_nat[6] * x_12_13_23_natur[0][2] + \
             B_nat[7] * x_123_natur[0]
y1_norm = B_norm[0] + B_norm[1] * X_matr[0][0] + B_norm[2] * X_matr[0][1] + B_norm[3] * X_matr[0][2] + B_norm[4] * \
              x_12_13_23[0][0] + B_norm[5] * x_12_13_23[0][1] + B_norm[6] * x_12_13_23[0][2] + B_norm[7] * x_123[0]

dx = [((X_max[i] - X_min[i]) / 2) for i in range(3)]
A = [sum(Y_average) / len(Y_average), B_nat[0] * dx[0], B_nat[1] * dx[1], B_nat[2] * dx[2]]

S_kv = [(sum([((Y_matr[i][j] - Y_average[i]) ** 2) for j in range(m)]) / m) for i in range(N)]

Gp = max(S_kv) / sum(S_kv)

f1 = m - 1
f2 = N


Gt = get_cohren_critical(p, f1, f2)

if Gp < Gt:
    print('Дисперсії однорідні')

S_average = sum(S_kv) / N

S2_beta_s = S_average / (N * m)

S_beta_s = S2_beta_s ** .5

beta = [(sum([x_for_beta[j][i] * Y_average[j] for j in range(N)]) / N) for i in range(4)]
ts = [(math.fabs(beta[i]) / S_beta_s) for i in range(4)]

f3 = f1 * f2

criterion_of_St = get_student_critical(p, f3)

result_2 = [criterion_of_Student(B_nat[0], criterion_of_St, ts[0]) +
                criterion_of_Student(B_nat[1], criterion_of_St, ts[1]) * X_matr_natur[i][0] +
                criterion_of_Student(B_nat[2], criterion_of_St, ts[2]) * X_matr_natur[i][1] +
                criterion_of_Student(B_nat[3], criterion_of_St, ts[3]) * X_matr_natur[i][2] for i in range(N)]

znach_koef = []
for i in ts:
    if i > criterion_of_St:
        znach_koef.append(i)
    else:
         pass

d = len(znach_koef)
f4 = N - d
f3 = (m - 1) * N

deviation_of_adequacy = (m / (N - d)) * sum([(result_2[i] - Y_average[i]) ** 2 for i in range(N)])

Fp = deviation_of_adequacy / S2_beta_s

Ft = get_fisher_critical(p, f3, f4)

print("Значення після критерія Стюдента:")
print("Y1 = {0:.3f};   Y2 = {1:.3f};   Y3 = {2:.3f};   Y4 = {3:.3f}.".format(result_2[0],
                                                                                 result_2[1],
                                                                                 result_2[2],
                                                                                 result_2[3]))
print("Y1a = {0:.3f};   Y2a = {1:.3f};   Y3a = {2:.3f};   Y4a = {3:.3f}.".format(Y_average[0],
                                                                                     Y_average[1],
                                                                                     Y_average[2],
                                                                                     Y_average[3]))

print(Ft)
if Fp < Ft:
    print('Fp = {} < Ft = {}'.format(round(Fp, 3), Ft))
    print('Рівняння регресії з ефектом взаємодії адекватно оригіналу при рівні значимості {}'.format(round(q, 2)))

else:
    print('Fp = {} > Ft = {}'.format(round(Fp, 3), Ft))
    print('Рівняння регресії з ефектом взаємодії неадекватно оригіналу при рівні значимості {}'.format(round(q, 2)))
    print("Додамо квадратичні коефіцієнти")

    N = 15
    l = 1.215

    Nt = []
    y1t = []
    y2t = []
    y3t = []
    ycp = []

    for i in range(1, 16):
        Nt.append(i)

    for i in range(len(Y_matr)):
        y1t.append(Y_matr[i][0])
        y2t.append(Y_matr[i][1])
        y3t.append(Y_matr[i][2])

    while len(y1t) != len(Nt):
        y1t.append(random.randint(Y_min, Y_max))
        y2t.append(random.randint(Y_min, Y_max))
        y3t.append(random.randint(Y_min, Y_max))

    for i in range(len(y1t)):
        ycp.append((y1t[i] + y2t[i] + y3t[i])/m)

    x1t = [-1, -1, -1, -1, 1, 1, 1, 1, -l, l, 0, 0, 0, 0, 0]
    x2t = [-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, -l, l, 0, 0, 0]
    x3t = [-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -l, l, 0]

    x1x2t = []
    x1x3t = []
    x2x3t = []
    x1x2x3t = []
    x1sqt = []
    x2sqt = []
    x3sqt = []

    for i in range(len(x1t)):
        x1x2t.append(x1t[i] * x2t[i])
        x2x3t.append(x2t[i] * x3t[i])
        x1x3t.append(x1t[i] * x3t[i])
        x1x2x3t.append(x1t[i] * x2t[i] * x3t[i])
        x1sqt.append(round(math.pow(x1t[i], 2), 3))
        x2sqt.append(round(math.pow(x2t[i], 2), 3))
        x3sqt.append(round(math.pow(x3t[i], 2), 3))

    tb = PrettyTable()
    tb.add_column("N", Nt)
    tb.add_column("x1", x1t)
    tb.add_column("x2", x2t)
    tb.add_column("x3", x3t)
    tb.add_column("x1x2", x1x2t)
    tb.add_column("x1x3", x1x3t)
    tb.add_column("x2x3", x2x3t)
    tb.add_column("x1x2x3", x1x2x3t)
    tb.add_column("x1^2", x1sqt)
    tb.add_column("x2^2", x2sqt)
    tb.add_column("x3^2", x3sqt)
    tb.add_column("y1", y1t)
    tb.add_column("y2", y2t)
    tb.add_column("y3", y3t)
    tb.add_column("y", ycp)

    print("Матриця планування експерименту для ОЦКП із нормованими значеннями факторів")
    print(tb)

    x01 = (x1min + x1max) / 2
    x02 = (x2min + x2max) / 2
    x03 = (x3min + x3max) / 2

    dx1 = x1max - x01
    dx2 = x2max - x02
    dx3 = x3max - x03

    x1 = [x1min, x1min, x1min, x1min, x1max, x1max, x1max, x1max, round(-l * dx1 + x01, 3), round(l * dx1 + x01, 3),
          x01, x01, x01, x01, x01]
    x2 = [x2min, x2min, x2max, x2max, x2min, x2min, x2max, x2max, x02, x02, round(-l * dx2 + x02, 3),
          round(l * dx2 + x02, 3), x02, x02, x02]
    x3 = [x3min, x3max, x3min, x3max, x3min, x3max, x3min, x3max, x03, x03, x03, x03, round(-l * dx3 + x03, 3),
          round(l * dx3 + x03, 3), x03]

    x1x2 = []
    x1x3 = []
    x2x3 = []
    x1x2x3 = []
    x1sq = []
    x2sq = []
    x3sq = []

    for i in range(len(Nt)):
        x1x2.append(round(x1[i] * x2[i], 3))
        x2x3.append(round(x2[i] * x3[i], 3))
        x1x3.append(round(x1[i] * x3[i], 3))
        x1x2x3.append(round(x1[i] * x2[i] * x3[i], 3))
        x1sq.append(round(math.pow(x1[i], 2), 3))
        x2sq.append(round(math.pow(x2[i], 2), 3))
        x3sq.append(round(math.pow(x3[i], 2), 3))

    tb1 = PrettyTable()
    tb1.add_column("N", Nt)
    tb1.add_column("x1", x1)
    tb1.add_column("x2", x2)
    tb1.add_column("x3", x3)
    tb1.add_column("x1x2", x1x2)
    tb1.add_column("x1x3", x1x3)
    tb1.add_column("x2x3", x2x3)
    tb1.add_column("x1x2x3", x1x2x3)
    tb1.add_column("x1^2", x1sq)
    tb1.add_column("x2^2", x2sq)
    tb1.add_column("x3^2", x3sq)
    tb1.add_column("y1", y1t)
    tb1.add_column("y2", y2t)
    tb1.add_column("y3", y3t)
    tb1.add_column("y", ycp)

    print("Матриця планування експерименту для ОЦКП із натуралізованими значеннями факторів")
    print(tb1)


    def countm(X):
        return sum(X) / len(X)


    def counta(X1, X2):
        s = 0
        for i in range(len(X1)):
            s += X1[i] * X2[i]
        return s / len(X1)


    X = [x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3, x1sq, x2sq, x3sq]
    x0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    X_for_beta = [x0, x1t, x2t, x3t, x1x2t, x1x3t, x2x3t, x1x2x3t, x1sqt, x2sqt, x3sqt]

    system = [[1], [], [], [], [], [], [], [], [], [], []]
    An = []

    An.append(countm(ycp))

    for i in range(len(X)):
        system[0].append(round(countm(X[i]), 3))
        system[i + 1].append(round(countm(X[i]), 3))
        An.append(round(counta(X[i], ycp), 3))

    A = [[], [], [], [], [], [], [], [], [], []]

    q = 0
    for i in range(len(A)):
        for j in range(q, len(X)):
            A[i].append(round(counta(X[i], X[j]), 3))
        q += 1

    k = 1
    for i in range(len(A)):
        system[i + 1].append(A[i][0])
        for j in range(1, len(A[i])):
            system[i + 1].append(A[i][j])
            system[j + k].append(A[i][j])
        k += 1

    M2 = np.array(system)
    v2 = np.array(An)

    result = np.linalg.solve(M2, v2)

    print("Коефіцієнти рівняння регресії:")

    for i in range(len(result)):
        print("b" + str(i) + " = " + str(round(result[i], 3)))

    print("Зробимо перевірку:")

    for i in range(len(x1)):
        print("y" + str(i + 1) + " = " + str(round(
            result[0] + result[1] * x1[i] + result[2] * x2[i] + result[3] * x3[i] + result[4] * x1x2[i] + result[5] *
            x1x3[i] + result[6] * x2x3[i] +
            + result[7] * x1x2x3[i] + result[8] * x1sq[i] + result[9] * x2sq[i] + result[10] * x3sq[i],
            3)) + " = " + str(ycp[i]))

    S = []

    print(str(result[0]) + " + " + str(result[1]) + " x1" + " + " + str(result[2]) + " x2" + " + " + str(
        result[3]) + " x3"
          + " + " + str(result[4]) + " x1x2" + " + " + str(result[5]) + " x1x3" + " + " + str(
        result[6]) + " x2x3" + " + " +
          str(result[7]) + " x1x2x3" + " + " + str(result[8]) + " x1^2" + " + " + str(result[9]) + " x2^2" + " + " +
          str(result[10]) + " x3^2")

    for i in range(15):
        S.append((math.pow((y1t[i] - ycp[i]), 2) + math.pow((y2t[i] - ycp[i]), 2) + math.pow((y3t[i] - ycp[i]), 2)) / 3)

    Gp = max(S) / sum(S)

    f1 = m - 1
    f2 = N

    Gt = get_cohren_critical(p, f1, f2)

    if Gp < Gt:
        print('Дисперсії однорідні')

    Sb = sum(S) / N
    Sbs2 = Sb / (N * m)
    Sbs = math.sqrt(Sbs2)

    B = []

    for i in range(len(X_for_beta)):
        B.append(counta(X_for_beta[i], ycp))

    f3 = f1 * f2

    criterion_of_St = get_student_critical(p, f3)

    d = 0
    for i in range(len(B)):
        if math.fabs(B[i]) / Sbs > criterion_of_St:
            print("Коефіцієнт b" + str(i) + "значимий")
            d += 1

        else:
            result[i] = 0

    ye = []
    for i in range(len(ycp)):
        yi = 0
        for j in range(len(result)):
            yi += result[j] * X_for_beta[j][i]
        ye.append(yi)

    f4 = N - d

    deviation_of_adequacy = (m / (N - d)) * sum([(ye[i] - ycp[i]) ** 2 for i in range(N)])

    Fp = deviation_of_adequacy / Sbs2
    Ft = get_fisher_critical(p, f3, f4)

    if Fp > Ft:
        print('Fp = {} > Ft = {}'.format(round(Fp, 3), Ft))
        print('Рівняння регресії неадекватно оригіналу при рівні значимості {}'.format(round(q, 2)))

    else:
        print('Fp = {} < Ft = {}'.format(round(Fp, 3), Ft))
        print('Рівняння регресії адекватно оригіналу при рівні значимості {}'.format(round(q, 2)))
        flag = False
