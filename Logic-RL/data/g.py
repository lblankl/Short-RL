global MaxACC
MaxACC = 0
class Graph:
    def __init__(self, Acc=0):
        global MaxACC
        print("Init MaxACC: ", MaxACC)
        if Acc > MaxACC:
            MaxACC = Acc
            print("After MaxACC: ", MaxACC)
        