import random
import numpy as np



def Algm(swarm_size):
    def generate(n, m, l, u):   # Generated solution
        data = []
        for i in range(n):
            tem = []
            for j in range(m):
                tem.append(random.uniform(l, u))
            data.append(tem)
        return data

    N, M, lb, ub = swarm_size, 5, 1, 5
    g, max_iter = 0, 100

    W = generate(N, M, lb, ub)

    def fitness(soln):
        F = []
        for i in range(len(soln)):
            F.append(random.random())
        return F


    def glob_exploration(W,F):
        Wnew=[]
        for i in range(len(W)):
            r1=random.uniform(0,1)
            Wnew.append(np.min(W[i])+r1*(np.max(W[i])-np.min(W[i])))

        W_=[]
        for i in range(len(W)):
            temp=[]
            for j in range(len(W[i])):
                if Wnew[j] <= np.mean(F[j]):
                    temp.append(Wnew[j])
                else:temp.append(W[i][j])
            W_.append(temp)

        return W_

    def Local_exploration(W,Wbest,F):
        W_new=[]
        for i in range(len(W)):
            temp=[]
            mi=10
            for j in range(len(W[i])):
                t=j+1
                T=max_iter
                sigma=((5*t/T)-2)/np.sqrt(25*np.square((5*t/T)-2))+0.7
                temp.append(Wbest[j]+((W[i][j]-Wbest[j])*(1+mi))/sigma)
            W_new.append(temp)
        W_=[]
        for i in range(len(W)):
            temp=[]
            for j in range(len(W[i])):
                if W_new[i][j] <= max(F):
                    temp.append(W_new[i][j])
                else:temp.append(W[i][j])
            W_.append(temp)

        return W_

    def Reflect_Electro_magnetic_waves(W):
        W_ = []
        for i in range(len(W)):
            temp = []
            for j in range(len(W[i])):
                epsilon=10^-6
                Alpha=0.45  # Step coefficient
                Wplus_epsilon=W[i][j]+epsilon
                Wminus_epsilon=W[i][j]-epsilon
                g=((Wplus_epsilon-Wminus_epsilon)/2*epsilon)
                temp.append(W[i][j]-Alpha*g)
            W_.append(temp)
        return W_

    def Received_Electromagnetic_waves(W):
        r2=random.uniform(0,1)
        r3=random.uniform(0,1)
        r4=random.uniform(0,1)
        r5=random.uniform(0,1)
        W_ = []
        for i in range(len(W)):
            temp = []
            for j in range(len(W[i])):
                t=j+1
                T=max_iter
                delta=0.6+(1.2-0.5)*np.sin((t*3.14)/(2*T))
                lambbda=(((2*t)/T)-0.7)/(0.78+abs((2*t)/T)-0.7)
                n=10
                eta=0.5
                if r5<=0.7:
                    temp.append(W[i][j]+delta*r2*(Wbest[j]-W[i][j])+eta*np.cos(3.14/n)*(Wbest[j]-W[i][j]))
                else:temp.append(W[i][j]+lambbda*r3*(Wbest[j]-W[i][j])+0.5*r4*(1-lambbda)*(Wbest[j]-W[i][j]))
            W_.append(temp)
        return W_

    while g<max_iter:
        Fit = fitness(W)
        bst=np.argmin(Fit)
        W1=glob_exploration(W, Fit)
        Wbest=W[bst]
        W2=Local_exploration(W1, Wbest,Fit)
        W3=Reflect_Electro_magnetic_waves(W2)
        W4=Received_Electromagnetic_waves(W3)
        bst_soln=W4[bst]

        g += 1

    return np.max(bst_soln)