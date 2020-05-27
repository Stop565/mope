import math
import random

from prettytable import PrettyTable

import numpy
from scipy.stats import f, t

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

p = 0.95
q = 1 - p

m = 3
N = 14

print("y = b0 + b1x1 + b2x2 + b3x3 + b12x1x2 + b13x1x3 + b23x2x3 + b123x1x2x3 + b11x1^2 + b22x2^2 + b33x3^2")

x1min = 15
x1max = 45
x2min = -70
x2max = 10
x3min = 15
x3max = 30

func = "9,1 + 3,9*x1 + 5,3*x2 + 4,6*x3 + 4,8*x1*x1 + 0,7*x2*x2 + 3,6*x3*x3 + 7,0*x1*x2 + 1,0*x1*x3 + 5,7*x2*x3 + 2,5*x1*x2*x3"

l = 1.73

Nt = []
for i in range(1, 15):
    Nt.append(i)

x1t = [-1, -1, -1, -1, 1, 1, 1, 1, -l, l, 0, 0, 0, 0]
x2t = [-1, -1, 1, 1, -1, -1, 1, 1, 0, 0, -l, l, 0, 0]
x3t = [-1, 1, -1, 1, -1, 1, -1, 1, 0, 0, 0, 0, -l, l]

flag = True
ct = 1 #кількість ітерацій
while (flag):
    flag = False
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

    print("Матриця планування експерименту для РЦКП із нормованими значеннями факторів")
    print(tb)

    x01 = (x1min + x1max) / 2
    x02 = (x2min + x2max) / 2
    x03 = (x3min + x3max) / 2

    dx1 = x1max - x01
    dx2 = x2max - x02
    dx3 = x3max - x03

    x1 = [x1min, x1min, x1min, x1min, x1max, x1max, x1max, x1max, round(-l * dx1 + x01, 3), round(l * dx1 + x01, 3),
          x01, x01, x01, x01]
    x2 = [x2min, x2min, x2max, x2max, x2min, x2min, x2max, x2max, x02, x02, round(-l * dx2 + x02, 3),
          round(l * dx2 + x02, 3), x02, x02]
    x3 = [x3min, x3max, x3min, x3max, x3min, x3max, x3min, x3max, x03, x03, x03, x03, round(-l * dx3 + x03, 3),
          round(l * dx3 + x03, 3)]

    x1x2 = []
    x1x3 = []
    x2x3 = []
    x1x2x3 = []
    x1sq = []
    x2sq = []
    x3sq = []

    y1t = []
    y2t = []
    y3t = []
    y4t = []
    yct = []

    for i in range(len(Nt)):
        x1x2.append(round(x1[i] * x2[i], 3))
        x2x3.append(round(x2[i] * x3[i], 3))
        x1x3.append(round(x1[i] * x3[i], 3))
        x1x2x3.append(round(x1[i] * x2[i] * x3[i], 3))
        x1sq.append(round(math.pow(x1[i], 2), 3))
        x2sq.append(round(math.pow(x2[i], 2), 3))
        x3sq.append(round(math.pow(x3[i], 2), 3))
        y1t.append(round(9.1 + 3.9 * x1[i] + 5.3 * x2[i] + 4.6 * x3[i] +  4.8 * x1sq[i] + 0.7 * x2sq[i] + 3.6 * x3sq[i]
                         + 7.0 * x1x2[i] + 1.0 * x1x3[i] + 5.7 * x2x3[i] + 2.5 * x1x2x3[i] + random.randint(0, 10) - 5,
                         3))
        y2t.append(round(9.1 + 3.9 * x1[i] + 5.3 * x2[i] + 4.6 * x3[i] +  4.8 * x1sq[i] + 0.7 * x2sq[i] + 3.6 * x3sq[i]
                         + 7.0 * x1x2[i] + 1.0 * x1x3[i] + 5.7 * x2x3[i] + 2.5 * x1x2x3[i] + random.randint(0, 10) - 5,
                         3))
        y3t.append(round(9.1 + 3.9 * x1[i] + 5.3 * x2[i] + 4.6 * x3[i] +  4.8 * x1sq[i] + 0.7 * x2sq[i] + 3.6 * x3sq[i]
                         + 7.0 * x1x2[i] + 1.0 * x1x3[i] + 5.7 * x2x3[i] + 2.5 * x1x2x3[i] + random.randint(0, 10) - 5,
                         3))
        y4t.append(round(9.1 + 3.9 * x1[i] + 5.3 * x2[i] + 4.6 * x3[i] +  4.8 * x1sq[i] + 0.7 * x2sq[i] + 3.6 * x3sq[i]
                         + 7.0 * x1x2[i] + 1.0 * x1x3[i] + 5.7 * x2x3[i] + 2.5 * x1x2x3[i] + random.randint(0, 10) - 5,
                         3))
        if m == 3:
            yct.append(round((y1t[i] + y2t[i] + y3t[i]) / 3, 3))
        else:
            yct.append(round((y1t[i] + y2t[i] + y3t[i] + y4t[i]) / 4, 3))

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
    tb1.add_column("y", yct)

    print("Матриця планування експерименту для РЦКП із натуралізованими значеннями факторів")
    print(tb1)


    def countm(X):
        return sum(X) / len(X)


    def counta(X1, X2):
        s = 0
        for i in range(len(X1)):
            s += X1[i] * X2[i]
        return s / len(X1)


    X = [x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3, x1sq, x2sq, x3sq]
    x0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    X_for_beta = [x0, x1, x2, x3, x1x2, x1x3, x2x3, x1x2x3, x1sq, x2sq, x3sq]

    system = [[1], [], [], [], [], [], [], [], [], [], []]
    An = []

    An.append(countm(yct))

    for i in range(len(X)):
        system[0].append(round(countm(X[i]), 3))
        system[i + 1].append(round(countm(X[i]), 3))
        An.append(round(counta(X[i], yct), 3))

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

    M2 = numpy.array(system)
    v2 = numpy.array(An)

    result = numpy.linalg.solve(M2, v2)

    print("Коефіцієнти рівняння регресії:")

    for i in range(len(result)):
        print("b" + str(i) + " = " + str(round(result[i], 3)))

    print("Зробимо перевірку:")

    for i in range(len(x1)):
        print("y" + str(i + 1) + " = " + str(round(
            result[0] + result[1] * x1[i] + result[2] * x2[i] + result[3] * x3[i] + result[4] * x1x2[i] + result[5] *
            x1x3[i] + result[6] * x2x3[i] +
            + result[7] * x1x2x3[i] + result[8] * x1sq[i] + result[9] * x2sq[i] + result[10] * x3sq[i],
            3)) + " = " + str(yct[i]))

    S = []

    print(str(result[0]) + " + " + str(result[1]) + " x1" + " + " + str(result[2]) + " x2" + " + " + str(
        result[3]) + " x3"
          + " + " + str(result[4]) + " x1x2" + " + " + str(result[5]) + " x1x3" + " + " + str(
        result[6]) + " x2x3" + " + " +
          str(result[7]) + " x1x2x3" + " + " + str(result[8]) + " x1^2" + " + " + str(result[9]) + " x2^2" + " + " +
          str(result[10]) + " x3^2")

    for i in range(N):
        S.append((math.pow((y1t[i] - yct[i]), 2) + math.pow((y2t[i] - yct[i]), 2) + math.pow((y3t[i] - yct[i]), 2)) / 3)

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
        B.append(counta(X_for_beta[i], yct))

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
    for i in range(len(yct)):
        yi = 0
        for j in range(len(result)):
            yi += result[j] * X_for_beta[j][i]
        ye.append(yi)

    f4 = N - d

    deviation_of_adequacy = (m / (N - d)) * sum([(ye[i] - yct[i]) ** 2 for i in range(N)])

    Fp = deviation_of_adequacy / Sbs2
    Ft = get_fisher_critical(p, f3, f4)

    if Fp > Ft:
        print('Fp = {} > Ft = {}'.format(round(Fp, 3), Ft))
        print('Рівняння регресії неадекватно оригіналу')
        m += 1
        flag = ct != 0
        ct -= 1

    else:
        print('Fp = {} < Ft = {}'.format(round(Fp, 3), Ft))
        print('Рівняння регресії адекватно оригіналу ')
        flag = False
