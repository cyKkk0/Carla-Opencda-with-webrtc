class receiver():
    def __init__(self, callback_func):
        self.call_back = callback_func

    def call_bk(self, x):
        self.call_back(x)

    def add_attr(self):
        self.x = 222      


def a(x):
    print(f'{x} I\'m a callback function!')

if __name__ == '__main__':
    recv = receiver(a)
    recv.add_attr()
    recv.call_bk(666)
    print(recv.x)