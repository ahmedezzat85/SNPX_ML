def fn(count):
    L = []
    for i in range(count):
        L.append(i+1)
    return i, L



i, a = fn(5)
a1, a2, a3, a4, a5 = a
print (i, a, a1, a2, a3, a4, a5)
