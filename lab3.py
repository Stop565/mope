import math
import random
import numpy as np
from prettytable import PrettyTable



def Cochran(F1, F2):
    table = [[0.9985, 0.9750, 0.9392, 0.9057, 0.8772, 0.8534, 0.8332, 0.8159, 0.8010, 0.7880],
             [0.9669, 0.8709, 0.7977, 0.7457, 0.7071, 0.6771, 0.6530, 0.6333, 0.6167, 0.6025],
             [0.9065, 0.7679, 0.6841, 0.6287, 0.5892, 0.5598, 0.5365, 0.5175, 0.5017, 0.4884],
             [0.8412, 0.6838, 0.5981, 0.5440, 0.5063, 0.4783, 0.4564, 0.4387, 0.4241, 0.4118],
             [0.7808, 0.6161, 0.5321, 0.4803, 0.4447, 0.4184, 0.3980, 0.3817, 0.3882, 0.3568],
             [0.7271, 0.5612, 0.4800, 0.4307, 0.3974, 0.3726, 0.3535, 0.3384, 0.3259, 0.3154],
             [0.6798, 0.5157, 0.4377, 0.3910, 0.3595, 0.3362, 0.3185, 0.3043, 0.2926, 0.2829],
             [0.6385, 0.4775, 0.4027, 0.3584, 0.3286, 0.3067, 0.2901, 0.2768, 0.2659, 0.2568],
             [0.6020, 0.4450, 0.3733, 0.3311, 0.3029, 0.2823, 0.2666, 0.2541, 0.2439, 0.2353]]
    return table[F2-2][F1-1]

def Student(F3):
    table = [12.71, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.261, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131]
    return table[F3 - 1]

def Fisher(F3, F4):
    table = [[164.4, 199.5, 215.7, 224.5, 230.2, 234.0],
             [18.5, 19.2, 19.2, 19.3, 19.3, 19.3],
             [10.1, 9.6, 9.3, 9.1, 9.0, 8.9],
             [7.7, 6.9, 6.6, 6.4, 6.3, 6.2],
             [6.6, 5.8, 5.4, 5.2, 5.1, 5.0],
             [6.0, 5.1, 4.8, 4.5, 4.4, 4.3],
             [5.5, 4.7, 4.4, 4.1, 4.0, 3.9],
             [5.3, 4.5, 4.1, 3.8, 3.7, 3.6],
             [5.1, 4.3, 3.9, 3.6, 3.5, 3.4],
             [5.0, 4.1, 3.7, 3.5, 3.3, 3.2]]
    return table[F3 - 1][F4 - 1]


print("y = b0 + b1x1 + b2x2 + b3x3")
tb = PrettyTable()
tb.add_column("№", [1, 2, 3, 4])
tb.add_column("x0", [1, 1, 1, 1])
tb.add_column("x1", [-1, -1, "+1", "+1"])
tb.add_column("x2", [-1, "+1", -1, "+1"])
tb.add_column("x0", [-1, "+1", "+1", -1])
print(tb)

m = 3
N = 4

x1min = 15
x1max = 45
x2min = -70
x2max = -10
x3min = 15
x3max = 30

xcmax = (x1max + x2max + x3max) // 3
xcmin = (x1min + x2min + x3min) // 3

ymax = 200 + xcmax
ymin = 200 + xcmin

x1 = [x1min, x1min, x1max, x1max]
x2 = [x2min, x2max, x2min, x2max]
x3 = [x3max, x3min, x3min, x3max]

Y1 = []
Y2 = []
Y3 = []

for i in range(4):
    Y1.append(random.randint(ymin, ymax))
    Y2.append(random.randint(ymin, ymax))
    Y3.append(random.randint(ymin, ymax))

tb1 = PrettyTable()
tb1.add_column("x1", x1)
tb1.add_column("x2", x2)
tb1.add_column("x3", x3)
tb1.add_column("y1", Y1)
tb1.add_column("y2", Y2)
tb1.add_column("y3", Y3)

print(tb1)

y1 = (Y1[0] + Y2[0] + Y3[0])/3
y2 = (Y1[1] + Y2[1] + Y3[1])/3
y3 = (Y1[2] + Y2[2] + Y3[2])/3
y4 = (Y1[3] + Y2[3] + Y3[3])/3

mx1 = (x1[0] + x1[1] + x1[2] + x1[3])/4
mx2 = (x2[0] + x2[1] + x2[2] + x2[3])/4
mx3 = (x3[0] + x3[1] + x3[2] + x3[3])/4

my = (y1 + y2 + y3 + y4)/4

a1 = (x1[0] * y1 + x1[1] * y2 + x1[2] * y3 + x1[3] * y4)/4
a2 = (x2[0] * y1 + x2[1] * y2 + x2[2] * y3 + x2[3] * y4)/4
a3 = (x3[0] * y1 + x3[1] * y2 + x3[2] * y3 + x3[3] * y4)/4

a11 = (x1[0] * x1[0] + x1[1] * x1[1] + x1[2] * x1[2] + x1[3] * x1[3])/4
a22 = (x2[0] * x2[0] + x2[1] * x2[1] + x2[2] * x2[2] + x2[3] * x2[3])/4
a33 = (x1[0] * x1[0] + x1[1] * x1[1] + x1[2] * x1[2] + x1[3] * x1[3])/4

a12 = a21 = (x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2] + x1[3] * x2[3])/4
a13 = a31 = (x1[0] * x3[0] + x1[1] * x3[1] + x1[2] * x3[2] + x1[3] * x3[3])/4
a23 = a32 = (x2[0] * x3[0] + x2[1] * x3[1] + x2[2] * x3[2] + x2[3] * x3[3])/4

b0m = np.array([[my, mx1, mx2, mx3], [a1, a11, a12, a13], [a2, a12, a22, a32], [a3, a13, a23, a33]])
b1m = np.array([[1, my, mx2, mx3], [mx1, a1, a12, a13], [mx2, a2, a22, a23], [mx3, a3, a23, a33]])
b2m = np.array([[1, mx1, my, mx3], [mx1, a11, a1, a13], [mx2, a12, a2, a23], [mx3, a13, a3, a33]])
b3m = np.array([[1, mx1, mx2, my], [mx1, a11, a12, a1], [mx2, a12, a22, a2], [mx3, a13, a23, a3]])
bm = np.array([[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a23], [mx3, a13, a23, a33]])

b0 = np.linalg.det(b0m) / np.linalg.det(bm)
b1 = np.linalg.det(b1m) / np.linalg.det(bm)
b2 = np.linalg.det(b2m) / np.linalg.det(bm)
b3 = np.linalg.det(b3m) / np.linalg.det(bm)

print(str(b0) + " + " + str(b1) + " x1 + " + str(b2) + " x2 + " + str(b3) + " x3")
print(str(y1) + " = " + str(b0 + b1 * x1[0] + b2 * x2[0] + b3 * x3[0]))
print(str(y2) + " = " + str(b0 + b1 * x1[1] + b2 * x2[1] + b3 * x3[1]))
print(str(y3) + " = " + str(b0 + b1 * x1[2] + b2 * x2[2] + b3 * x3[2]))
print(str(y4) + " = " + str(b0 + b1 * x1[3] + b2 * x2[3] + b3 * x3[3]))

S = []
S.append((math.pow((Y1[0] - y1), 2) + math.pow((Y2[0] - y1), 2) + math.pow((Y3[0] - y1), 2))/3)
S.append((math.pow((Y1[1] - y2), 2) + math.pow((Y2[1] - y2), 2) + math.pow((Y3[1] - y2), 2))/3)
S.append((math.pow((Y1[2] - y3), 2) + math.pow((Y2[2] - y3), 2) + math.pow((Y3[2] - y3), 2))/3)
S.append((math.pow((Y1[3] - y4), 2) + math.pow((Y2[3] - y4), 2) + math.pow((Y3[3] - y4), 2))/3)

Gp = max(S)/(S[0] + S[1] + S[2] + S[3])
f1 = m - 1
f2 = N

if Gp < Cochran(f1,f2):
    print("\nДисперсія однорідна. Рівень значимості = 0.05")

print(Cochran(f1,f2))

Sb = (S[0] + S[1] + S[2] + S[3])/N
Sbs2 = Sb/(N * m)
Sbs = math.sqrt(Sbs2)

x0s = [1, 1, 1, 1]
x1s = [-1, -1, 1, 1]
x2s = [-1, 1, -1, 1]
x3s = [-1, 1, 1, -1]

B0 = (y1 * x0s[0] + y2 * x0s[1] + y3 * x0s[2] + y4 * x0s[3])/N
B1 = (y1 * x1s[0] + y2 * x1s[1] + y3 * x1s[2] + y4 * x1s[3])/N
B2 = (y1 * x2s[0] + y2 * x2s[1] + y3 * x2s[2] + y4 * x2s[3])/N
B3 = (y1 * x3s[0] + y2 * x3s[1] + y3 * x3s[2] + y4 * x3s[3])/N

t0 = math.fabs(B0)/Sbs
t1 = math.fabs(B1)/Sbs
t2 = math.fabs(B2)/Sbs
t3 = math.fabs(B3)/Sbs

print(t0)
print(t1)
print(t2)
print(t3)

f3 = f1 * f2

d = 0

Y1e = []
Y2e = []
Y3e = []
Y4e = []

if t0 > Student(f3):
    print("\nКоефієнт b0 значний")
    d += 1
    Y1e.append(b0)
    Y2e.append(b0)
    Y3e.append(b0)
    Y4e.append(b0)

if t1 > Student(f3):
    print("\nКоефієнт b1 значний")
    d += 1
    Y1e.append(b1 * x1[0])
    Y2e.append(b1 * x1[1])
    Y3e.append(b1 * x1[2])
    Y4e.append(b1 + x1[3])

if t2 > Student(f3):
    print("\nКоефієнт b2 значний")
    d += 1
    Y1e.append(b2 * x2[0])
    Y2e.append(b2 * x2[1])
    Y3e.append(b2 * x2[2])
    Y4e.append(b2 + x2[3])

if t3 > Student(f3):
    print("\nКоефієнт b3 значний")
    d += 1
    Y1e.append(b3 * x3[0])
    Y2e.append(b3 * x3[1])
    Y3e.append(b3 * x3[2])
    Y4e.append(b3 + x3[3])

y1e = 0
y2e = 0
y3e = 0
y4e = 0

for i in range(len(Y1e)):
    y1e += Y1e[i]

for i in range(len(Y2e)):
    y2e += Y2e[i]

for i in range(len(Y3e)):
    y3e += Y3e[i]

for i in range(len(Y4e)):
    y4e += Y4e[i]

print(str(y1) + " = " + str(y1e))
print(str(y2) + " = " + str(y2e))
print(str(y3) + " = " + str(y3e))
print(str(y4) + " = " + str(y4e))

Sad = m/(N - d) * (pow((y1e - y1), 2) + pow((y2e - y2), 2) + pow((y3e - y3), 2) + pow((y4e - y4), 2))

Fp = Sad/Sbs2

f4 = N - d

if Fp > Fisher(f3, f4):
    print("\nРівняння регресії адекватно оригіналу при рівні значимості 0.05")
