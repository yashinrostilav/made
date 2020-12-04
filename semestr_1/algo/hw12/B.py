from sys import stdin


class A:
    def __init__(self):
        self.p = None
        self.r = None
        self.v = None
        self.child = None

    def init(self, n):
        self.p = [None] * n
        self.child = [None] * n
        self.r = ([1] * n).copy()
        self.v = ([0] * n).copy()
        for i in range(n):
            self.p[i] = i
            self.child[i] = [i]

    def get(self, x):
        if self.p[x] != x:
            self.p[x] = self.get(self.p[x])
        return self.p[x]

    def fun(self, x, y):
        x = self.get(x)
        y = self.get(y)
        if x != y:
            if self.r[x] < self.r[y]:
                x, y = y, x
            self.p[y] = x
            self.r[x] = self.r[x] + self.r[y]
            for _ in self.child[y]:
                self.child[x].append(_)

    def exp(self, x, v):
        x = self.get(x)
        for _ in self.child[x]:
            self.v[_] += v

    def out(self, x):
        return self.v[x]

def main():
    for line in stdin.buffer.read().decode().splitlines():
        inpt = list(map(str, line.split()))
        oper = inpt[0]
        if oper == 'join':
            cl.fun(int(inpt[1]),int(inpt[2]))
        elif oper == 'add':
            cl.exp(int(inpt[1]),int(inpt[2]))
        elif oper == 'get':
            print(cl.out(int(inpt[1])))
        else:
            n = int(oper)
            cl = A()
            cl.init(n + 1)

if __name__ == '__main__':
    main()

