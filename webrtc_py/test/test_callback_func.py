class receiver():

    def call_bk(self, x):
        self.call_back(x)

    def add_attr(self):
        self.x = 222      


def a(x):
    print(f'{x} I\'m a callback function!')

if __name__ == '__main__':
    recv = receiver()
    recv.call_back = a
    recv.add_attr()
    recv.call_bk(666)
    print(recv.x)