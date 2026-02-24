import numpy as np


def objective_function(x):
    # Example: Sphere function
    return np.sum(x ** 2)



def TSO(num_agents, dim, max_iter, lb, ub):
    # --- Equation (1) Initialization ---
    X = lb + np.random.rand(num_agents, dim) * (ub - lb)

    # Evaluate fitness
    fitness = np.array([objective_function(x) for x in X])
    best_idx = np.argmin(fitness)
    X_best = X[best_idx].copy()
    f_best = fitness[best_idx]

    # --- Iterations ---
    for t in range(max_iter):
        # --- Equation (2) Transition Factor ---
        TF = np.sin((np.pi * t) / (2 * max_iter))

        for i in range(num_agents):
            # Choose two random agents j, k (different from i)
            idx = list(range(num_agents))
            idx.remove(i)
            j, k = np.random.choice(idx, 2, replace=False)

            # --- Equation (3) Velocity Update ---
            rand1 = np.random.rand(dim)
            rand2 = np.random.rand(dim)

            V_i = (TF * (rand1 * (X_best - X[i]))) + ((1 - TF) * rand2 * (X[j] - X[k]))

            # --- Equation (4) Position Update ---
            X_new = X[i] + V_i

            # --- Equation (5) Boundary Check ---
            X_new = np.clip(X_new, lb, ub)

            # Evaluate new fitness
            f_new = objective_function(X_new)

            # Greedy selection
            if f_new < fitness[i]:
                X[i] = X_new
                fitness[i] = f_new

                # Update global best
                if f_new < f_best:
                    X_best = X_new.copy()
                    f_best = f_new

        # Print iteration info
        print(f"Iter {t + 1}/{max_iter}, Best Fitness = {f_best:.6f}")

    return X_best, f_best


def Algm(swarm_size):

    dim = 5  # Problem dimension
    max_iter = 50  # Number of iterations
    lb, ub = -10, 10  # Search space bounds

    best_sol, best_fit = TSO(swarm_size, dim, max_iter, lb, ub)

    return np.max(best_sol)