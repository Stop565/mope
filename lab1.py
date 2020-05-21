import random
from prettytable import PrettyTable





tb = PrettyTable()
tb.add_column("№", [1, 2, 3, 4, 5, 6, 7, 8, "X0", "dx"])

a = [1, 2, 3, 4]
x1 = []
x2 = []
x3 = []
y = []

for i in range(8):
    x1.append(random.randint(0, 20))
    x2.append(random.randint(0, 20))
    x3.append(random.randint(0, 20))
    y.append(a[0] + a[1] * x1[i] + a[2] * x2[i] + a[3] * x3[i])

x1.append((max(x1) + min(x1)) / 2)
x2.append((max(x2) + min(x2)) / 2)
x3.append((max(x3) + min(x3)) / 2)
y.append("")

x1.append(x1[len(x1) - 1] - min(x1))
x2.append(x2[len(x2) - 1] - min(x2))
x3.append(x3[len(x3) - 1] - min(x3))
y.append("")

tb.add_column("X1", x1)
tb.add_column("X2", x2)
tb.add_column("X3", x3)
tb.add_column("Y", y)

xn1 = []
xn2 = []
xn3 = []

for i in range(8):
    xn1.append((x1[i] - x1[len(x1) - 2]) / x1[len(x1) - 1])
    xn2.append((x2[i] - x2[len(x2) - 2]) / x2[len(x2) - 1])
    xn3.append((x3[i] - x3[len(x3) - 2]) / x3[len(x3) - 1])

for i in range(2):
    xn1.append("")
    xn2.append("")
    xn3.append("")

tb.add_column("Xн1", xn1)
tb.add_column("Xн2", xn2)
tb.add_column("Xн3", xn3)

print(tb)

ye = a[0] + a[1] * x1[len(x1) - 2] + a[2] * x2[len(x2) - 2] + a[3] * x3[len(x3) - 2]

possible = []

for i in range(len(y) - 2):
    if float(y[i]) > ye:
        possible.append(y[i])

print(str(ye) + " < " + str(min(possible)))
