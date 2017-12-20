def fn(a=None, b=None):
    print ('fn', a, b)

def f1(x, c=10, **kwargs):
    print ('f1  ', c)
    fn(**kwargs)

d = {'a': 17, 'b': 88}
f1(7, **d, c=12)