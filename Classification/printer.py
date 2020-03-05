class printer():
    def __init__(self):
        self.yes = 1
        self.no = 2
        self.maybe = 4
        self.definitely = 8
        self.definitelynot = 16

    def printy(self, params):
        if (params & self.yes):
            print("Yes")

        if (params & self.no):
            print("No")

        if (params & self.maybe):
            print("Maybe")

        if (params & self.definitely):
            print("Definitely")

        if (params & self.definitelynot):
            print("Definitely Not")

mp = printer()
mp.printy(mp.yes | mp.no | mp.definitelynot)