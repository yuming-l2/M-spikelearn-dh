import time

f=open('output_mnist_1t1r_v.txt')
content=[line.split() for line in f if line.startswith('STDPC ')]

def isgood(fields):
    if not any([float(x)>=float(fields[4])-0.051 for x in fields[7:][-10:]]):
        return False
    if not any([float(x)>=float(fields[4])-0.101 for x in fields[7:][-5:]]):
        return False
    if int(fields[6])>5 and not any(float(x)>=0.20 for x in fields[7:][-3:]):
        return False
    if int(fields[6])>3 and not any(float(x)>=0.15 for x in fields[7:][-2:]):
        return False
    return True

A={(fields[1],fields[2],fields[3]):(' '.join(fields[7:][-1:])+f' {float(fields[4]):.3f} {int(fields[5]):2d} {int(fields[6]):2d}  {"o" if isgood(fields) else "x"}') for fields in content}

while True:
    try:
        L=input()
        time.sleep(0.1)
        p1, centerW, p2,p3, Alrp, Alrn = L.split()
        p1=float(p1)
        p2=float(p2)
        p3=float(p3)
        print(p1, centerW, Alrp, Alrn, A[('%.6f'%p1, '%.4f'%p2, '%.4f'%p3)])
    except EOFError as e:
        raise e
    except Exception as e:
        print(type(e).__name__, str(e))
