class Origin:
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return "Origin" + str(self.id)


class Destination:
    def __init__(self, id):
        self.id = id

    def __repr__(self):
        return "Destination" + str(self.id)
