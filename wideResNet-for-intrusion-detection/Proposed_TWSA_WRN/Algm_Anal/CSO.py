import random, numpy as np
def Algm(S):
    def initialize_soln(N, M):  # solution
        soln = []
        for i in range(N):  # N rows, M columns
            tem = []
            for j in range(M): tem.append(random.random())  # random values
            soln.append(tem)
        return soln

    def initialize_velo(N, M):  # velocity
        velo = []
        for i in range(N):  # N rows, M columns
            tem = []
            for j in range(M): tem.append(random.random())  # random values
            velo.append(tem)
        return velo

    def fitness(solution): # fitness calculation
        fit = []
        for i in range(len(solution)):
            sum = 0
            for j in range(len(solution[i])):
                sum += solution[i][j] # summation of soln.
            fit.append(sum)
        return fit

    def initialize_SPC(N): # self-position consideration (SPC)
        spc=[]
        for i in range(N):
            spc.append(random.randint(0,1))
        return spc

    def initialize_prev(N, M): # previous solution
        ps = []
        for i in range(N):
            tem = []
            for j in range(M):
                tem.append(0)
            ps.append(tem)
        return ps

    def seeking_process(ps,soln): # Seeking update
        soln_update = []
        for j in range(len(soln)):
            SRD = abs(soln[j]-ps[j])
            R = random.uniform(-1,1)
            soln_update.append(abs(1.0+(SRD*R))*soln[j])
        return soln_update

    def probability_calc(Fit, FSi): # probability calculation
        FSb = min(Fit)
        FSmax = max(Fit)
        FSmin = min(Fit)
        return ((FSi-FSb)/(FSmax-FSmin))

    def velocity_update(velocity, Xbest, X): # velocity update
        new_velo = []
        c1 = 1.0
        r1 = random.uniform(0,1)
        for i in range(len(velocity)):
            tem = []
            for j in range(len(velocity[i])):
                tem.append(abs(velocity[i][j]+(r1*c1*(Xbest[j]-X[i][j]))))  # velocity update
            new_velo.append(tem)
        return new_velo

    def tracing_process(X,V): # Tracing update
        soln_update = []
        for j in range(len(X)):
            soln_update.append(X[j]+V[j])
        return soln_update

    # Initializing parameters
    N , M, Tmax, g = int(S),5,10, 0 # rows(cats), column size, maximum iteration
    solution = initialize_soln(N, M)
    velocity = initialize_velo(N, M)
    SPC = initialize_SPC(N)
    Fit = []
    BEST_SOLUTION = []
    overall_best = []
    overall_fit = []
    prev_solution = initialize_prev(N, M)

    # Main loop
    while (g < Tmax):
        new_solution = []
        Fit = fitness(solution) # fitness calculation
        best = np.argmin(Fit) # minimization problem
        overall_fit.append(min(Fit))
        overall_best.append(solution[best])

        SMP = solution.copy() # seeking memory pool
        velocity = velocity_update(velocity, solution[best], solution)

        # for each row of solution, check SPC
        for i in range(len(solution)):
            # Seeking mode
            if (SPC[i] == 1):
                new_solution.append(seeking_process(prev_solution[i], solution[i]))
                Pi = probability_calc(Fit, Fit[i]) # probability of current cat (soln.)
            #Tracing mode
            else:
                new_solution.append(tracing_process(solution[i], velocity[i]))
        prev_solution = solution.copy()
        solution = new_solution.copy() # updated with new soln.

        g = g+1
    best = np.argmin(overall_fit) # minimization
    BEST_SOLUTION = overall_best[best]

    return np.max(BEST_SOLUTION)

