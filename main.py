import numpy as np

# ROMAN NUMERALS
# I = 1
# V = 5
# X = 10
# L = 50
# C = 100
# D = 500
# M = 1000


# SPECIAL CASES:
# if 4 * 10 ** n, then Roman Numeral is written as ROMAN_NUMERALS[n][0] + ROMAN_NUMERALS[n+1][0]
# if 8 * 10 ** n, then Roman Numeral is written as ROMAN_NUMERALS[n][0] + ROMAN_NUMERALS[n][0] + ROMAN_NUMERALS[n+1][0]
# if 9 * 10 ** n, then Roman Numeral is written as ROMAN_NUMERALS[n][0] + ROMAN_NUMERALS[n+1][0]


X = 492

ROMAN_NUMERALS = (("I", 1), ("V", 5), ("X", 10), ("L", 50), ("C", 100), ("D", 500), ("M", 1000))

y = np.zeros(4, dtype=int)

Xtemp = X

for j in range(4):
    y[j] = int(Xtemp / (1000 / 10 ** j))
    Xtemp = Xtemp - y[j] * (1000 / 10 ** j)

for i in range(len(ROMAN_NUMERALS)):
    if X / ROMAN_NUMERALS[i][1] < 1:
        index = i - 1
        break

print("kolejne liczby w X: ", y)
print("najwieksza liczba rzymska w zapisie: ", ROMAN_NUMERALS[index][0])




