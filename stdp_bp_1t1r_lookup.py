import time

f=open('output.txt')
content=[line.split() for line in f if line.startswith('t ')]

def isgood(fields):
    return any([float(x)>=float(fields[4])-0.051 for x in fields[7:][-10:]]) and any([float(x)>=float(fields[4])-0.101 for x in fields[7:][-5:]])

A={(fields[1],fields[2],fields[3]):(f'{float(fields[4]):.3f} {int(fields[5]):2d} {int(fields[6]):2d}  {" " if isgood(fields) else "x"} '+' '.join(fields[7:])) for fields in content}

while True:
    try:
        L=input()
        time.sleep(0.1)
        pid,_,__,___,____,_____,p1,p2,p3 = L.split()
        p1=float(p1)
        p2=float(p2)
        p3=float(p3)
        print(pid, A[('%.6f'%p1, '%.4f'%p2, '%.4f'%p3)])
    except EOFError as e:
        raise e
    except Exception as e:
        print(type(e).__name__, str(e))
