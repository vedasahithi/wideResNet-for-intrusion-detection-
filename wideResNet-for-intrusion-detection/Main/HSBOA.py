import random,numpy as np

def Algm():

    def generate(r,c,l,u):
        Data=[]
        for row in range(r):
            tem=[]
            for column in range(c):
                tem.append(random.uniform(l,u))
            Data.append(tem)
        return Data

    N,M,Lb,Ub =10,5,0,1
    g, max_it=0,100
    soln=generate(N,M,Lb,Ub)

    def fitness(soln):     #objective function
        F=[]
        for i in range(len(soln)):
            F.append(random.random())
        return F

    def hunting_strategy_secretary_bird(X,f):  # Exploration phase
        Xnew=[]
        for i in range(len(X)):
            temp=[]
            xrand1=random.random()
            xrand2=random.random()
            R1=10
            for j in range(len(X[i])):
                temp.append(X[i][j]+(xrand1-xrand2)*R1)
            Xnew.append(temp)

        f1=fitness(Xnew)
        X_=[]
        for i in range(len(X)):
            temp=[]
            for j in range(len(X[i])):
                if f1[j]<f[j]:
                    temp.append(Xnew[i][j])
                else:temp.append(X[i][j])
            X_.append(temp)

        return X_

    def stage2(xbest,X,f):
        RB=random.randint(1,M)
        Xnew=[]
        for i in range(len(X)):
            temp=[]
            T=max_it
            for j in range(len(X[i])):
                t=j+1
                temp.append(xbest[j]+np.exp(t/T)*(RB-0.5)*(xbest[j]-X[i][j]))
            Xnew.append(temp)
        X_=[]
        f1 = fitness(Xnew)
        for i in range(len(X)):
            temp=[]
            for j in range(len(X[i])):
                if f1[j]<f[j]:
                    temp.append(Xnew[i][j])
                else:
                    temp.append(X[i][j])
            X_.append(temp)

        return X_


    def stage3(xbest,X,f):  # Attacking prey
        Levy=10
        RL=0.5*Levy
        Xnew = []
        for i in range(len(X)):
            temp = []
            T = max_it
            for j in range(len(X[i])):
                t = j + 1
                temp.append(xbest[j] + ((1-(t/T)*(2*(t/T))))*X[i][j]*RL)
            Xnew.append(temp)
        X_ = []
        f1 = fitness(Xnew)
        for i in range(len(X)):
            temp = []
            for j in range(len(X[i])):
                if f1[j] < f[j]:
                    temp.append(Xnew[i][j])
                else:
                    temp.append(X[i][j])
            X_.append(temp)

        return X_

    def escape_strategy_secretarybird(Xbest,X,f):  # Exploitation stage
        C1=5   #Camouflage by environment
        C2=10  #Fly or run away
        ri=random.uniform(0,1)
        r=0.5
        R2=random.randint(1,M)
        K=round(1+random.uniform(1,1))
        RB = random.randint(1, M)
        Xnew=[]
        for i in range(len(X)):
            temp = []
            T = max_it
            Xrand=random.random()
            for j in range(len(X[i])):
                t = j + 1
                if r<ri:
                    temp.append(C1*Xbest[j]+(2*RB-1)*np.square(1-(t/T)))
                else:
                    temp.append((1/2*np.cos(np.pi*t)-1)*((R2*Xrand)*(1+2*np.cos(np.pi*t)-(X[i-1][j-1]*(1-2*np.cos(np.pi*t))))*(1-R2*K)))  # updated equation
            Xnew.append(temp)

        X_ = []
        f1 = fitness(Xnew)
        for i in range(len(X)):
            temp = []
            for j in range(len(X[i])):
                if f1[j] < f[j]:
                    temp.append(Xnew[i][j])
                else:
                    temp.append(X[i][j])
            X_.append(temp)

        return X_



    while g<max_it:
        fit = fitness(soln)
        bst=np.argmax(fit)
        xbest=soln[bst]
        X=hunting_strategy_secretary_bird(soln, fit)

        X=stage2(xbest, X, fit)

        X=stage3(xbest, X, fit)
        X_upd=escape_strategy_secretarybird(xbest, X, fit)
        bst_soln=X_upd[bst]
        g += 1

    return np.max(bst_soln)
