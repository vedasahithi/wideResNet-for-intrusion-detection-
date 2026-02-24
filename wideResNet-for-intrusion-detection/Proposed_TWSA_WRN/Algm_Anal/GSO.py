import random,math
import numpy as np
def Algm(swarm_size):
    def generate(n, m, l, u):   #Initial position
        data = []
        for i in range(n):
            tem = []
            for j in range(m):
                tem.append(random.uniform(l, u))
            data.append(tem)
        return data

    N, M, lb, ub = swarm_size, 5, 1, 5
    g, max = 0, 100

    soln = generate(N, M, lb, ub)


    def fitness(soln):
        F = []
        for i in range(len(soln)):
            F.append(random.random())
        return F
    gBest, gfit = [], float('inf')

    def Think_decision_Move(location,rate):
        Alpha=random.random()
        N_L=[]
        for i in range(len(location)):
            temp=[]
            for j in range(len(location[i])):
                if Alpha> random.random():md=1
                else:md=-1
                temp.append(location[i][j]+md*(rate[j]-rate[j-1])*(location[i][j]-location[i-1][j-1])*random.random())
            N_L.append(temp)
        return N_L


    def Correct_Location(location,new_location):
        beta=random.random()
        gamma=random.random()
        CL=[]
        for i in range(len(location)):
            temp=[]
            for j in range(len(location[i])):
                if random.random() < beta:
                    temp.append(location[i-1][j-1])  # neighbour location
                elif beta < random.random() < gamma:
                    temp.append(new_location[i][j])  # New location
                else:
                    temp.append(location[i][j])  # Donot move
                CL.append(temp)
        return CL

    while g<max:
        fit = fitness(soln)
        bst = np.argmin(fit)
        new_location=Think_decision_Move(soln,fit)
        crt_location=Correct_Location(soln,new_location)
        Bst_Soln = crt_location[bst]

        g+=1
    return np.max(Bst_Soln)