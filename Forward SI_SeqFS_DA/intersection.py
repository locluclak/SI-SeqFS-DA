import numpy as np
from bisect import bisect, bisect_left, bisect_right
class Obj:
    def __init__(self, v, bracket: bool, id = 0):
        """
        False: ( 
        True: )
        """
        self.value = v
        self.ooc = bracket
        self.id = id
    def __lt__(self, other):
        if self.value < other.value:
            return True
        return False

    def show(self):
        if self.ooc == False:
            print(f"({self.value}", end=" ")
        else:
            print(f"{self.value})", end=" ")


def solve_quadratic_inequality(a, b, c,seed = 0):
    """ ax^2 + bx +c <= 0 """
    if abs(a) < 1e-8:
        a = 0
    if abs(b) < 1e-8:
        b = 0
    if abs(c) < 1e-8:
        c = 0
    if a == 0:
        # print(f"b: {b}")
        if b > 0:
            # return [(-np.inf, -c / b)]
            return [(-np.inf, np.around(-c / b, 8))]
        elif b == 0:
            # print(f"c: {c}")
            if c <= 0:
                return [(-np.inf, np.inf)]
            else:
                print('Error bx + c', seed)
                return 
        else:
            return [(np.around(-c / b, 8), np.inf)]
    delta = b*b - 4*a*c
    if delta < 0:
        if a < 0:
            return [(-np.inf, np.inf)]
        else:
            print("Error to find interval. ")
    # print("delta:", delta)
    # print(f"2a: {2*a}")
    x1 = (- b - np.sqrt(delta)) / (2.0*a)
    x2 = (- b + np.sqrt(delta)) / (2.0*a)
    # if x1 > x2:
    #     x1, x2 = x2, x1  
    x1 = np.around(x1, 8)
    x2 = np.around(x2, 8)
    if a < 0:
        return [(-np.inf, x2),(x1, np.inf)]
    return [(x1,x2)]

def Intersec(a: list, b: list) -> list:
    Raxis = []
    for interval in a:
        l, r = interval
        Raxis.append(Obj(l, False, 0))
        Raxis.append(Obj(r, True, 0))
    for interval in b:
        l, r = interval
        l_ = Obj(l, False, 1)
        r_ = Obj(r, True, 1)
        Raxis.insert(bisect_left(Raxis, l_), l_)
        Raxis.insert(bisect_left(Raxis, r_), r_)
    
    finalinterval = []
    stack = []
    on0 = False
    on1 = False
    for obj in Raxis:
        # print('||| intersec')
        # obj.show()
        # print()


        if obj.ooc == False:
            if obj.id == 0:
                on0 = True 
            if obj.id == 1:
                on1 = True
            if len(stack) != 0:
                stack.pop()
            stack.append(obj)
        else:
            if len(stack) == 0:
                if obj.id == 0:
                    on0 = False 
                if obj.id == 1:
                    on1 = False
                continue
            temp = stack.pop()
            if on0 and on1:
                finalinterval.append((temp.value, obj.value))
            if obj.id == 0:
                on0 = False 
            if obj.id == 1:
                on1 = False
    return finalinterval
def Union(a: list, b: list) -> list:
    Raxis = []
    for interval in a:
        l, r = interval
        Raxis.append(Obj(l, False))
        Raxis.append(Obj(r, True))
    for interval in b:
        l, r = interval
        l_ = Obj(l, False)
        r_ = Obj(r, True)
        Raxis.insert(bisect_left(Raxis, l_), l_)
        Raxis.insert(bisect_left(Raxis, r_), r_)
    
    finalinterval = []
    stack = []
    for obj in Raxis:
        if obj.ooc == False:
            stack.append(obj)
        else:
            temp = stack.pop()
            if len(stack) == 0:
                finalinterval.append((temp.value, obj.value))
    
    return finalinterval

if __name__ == "__main__":
    da = [(-0.4122605383393618, -0.3804600576589586)] 
    fs = [(-0.47042270208769704, -0.3817176404989392)] 
    aic= [(-1.5756484596917149, -0.407189815328346), (1.0623004540292729, 9.392823837961563)]
# intervalinloop: [(-0.4122605383393618, -0.407189815328346), (1.0623004540292729, 9.392823837961563)]
    dafs = Intersec(da, fs)
    dafsaic = Intersec(dafs, aic)
    print(dafsaic)