def hello2(a=0,b=0):
    print('hello2:', a, b)


def hello1(a=0,b=0):
    print('hello1:', a, b)
    hello2(a=a+1,b=b+1)


if __name__=='__main__':
    hello1(a=1,b=1)