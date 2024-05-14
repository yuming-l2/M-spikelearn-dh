import time

f=open('output_mnist_1t1r_v.txt')
content=[line.split() for line in f if line.startswith('STDPV ')]

def isgood(fields):
    if not any([float(x)>=float(fields[4])-0.051 for x in fields[7:][-10:]]):
        return False
    if not any([float(x)>=float(fields[4])-0.101 for x in fields[7:][-5:]]):
        return False
    if int(fields[6])>5 and not any(float(x)>=0.20 for x in fields[7:][-3:]):
        return False
    if int(fields[6])>3 and not any(float(x)>=0.15 for x in fields[7:][-2:]):
        return False
    if int(fields[6])>=3 and float(fields[8])<0.3 and float(fields[9])<0.3 or int(fields[6])>=4 and float(fields[10])<0.4:
        return False
    return True

A={(fields[1],fields[2],fields[3]):(f'{float(fields[4]):.3f} {int(fields[5]):2d} {int(fields[6]):2d}  {" " if isgood(fields) else "x"} '+' '.join(fields[7:])) for fields in content}

while True:
    try:
        L=input()
        time.sleep(0.1)
        pid,_u,_a,_e,_f,_dv,p1,p2,p3,_sl1,_sl2 = L.split()
        p1=float(p1)
        p2=float(p2)
        p3=float(p3)
        print(pid, A[('%.6f'%p1, '%.4f'%p2, '%.4f'%p3)])
    except EOFError as e:
        raise e
    except Exception as e:
        print(type(e).__name__, str(e))
