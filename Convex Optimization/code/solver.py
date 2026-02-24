import numpy as np
import argparse
import time
# import scipy

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='input file path')
    parser.add_argument('path_output', help='output file path')
    args = parser.parse_args()
    return args.path_input, args.path_output
    
def read_problem(path):
    f = open(file=path, mode='rt')
    first_row = f.readline()
    n = int(first_row.split(' ')[0])
    W = int(first_row.split(' ')[1])
    pairs = []
    for i in range(n):
        string_pair = f.readline()
        p_k = int(string_pair.split(' ')[0])
        w_k = int(string_pair.split(' ')[1])
        pairs.append([p_k, w_k])
    f.close()
    pairs = np.array(pairs)
    return n, W, pairs
    
def write_results(path, z_star, x_star):
    f = open(file=path, mode='w')
    f.write(f'{int(z_star)}\n')
    for i in range(x_star.shape[0]):
        f.write(f'{int(x_star[i])}\n')
    f.close()
    
class KnapsackProblem:

    def __init__(self, n:int, W:int, pairs:np.array):

        if n is  None:
           raise Exception('n = None')
        if W is None:
            raise Exception('W = None')
        if pairs is None:
            raise Exception('pairs = None')
        if n != pairs.shape[0] or pairs.shape[1] != 2:
            raise Exception('Invalid \'n\' or \'pairs\' shape')
        if W < 0:
            raise Exception('W < 0')
        
        self.__n = n
        self.__W = W
        self.__pairs = np.copy(pairs)  
        
    @property
    def n(self):
        return self.__n
    @property
    def W(self):
        return self.__W
    @property
    def pairs(self):
        return np.copy(self.__pairs)
        
class KnapsackProblemSolver(KnapsackProblem):  
    
    def __init__(self, problem:KnapsackProblem):
    
        if problem is None:
            raise Exception('problem = None')
    
        super().__init__(problem.n, problem.W, problem.pairs)
        
        self.__relative_profits = self.__relative_profits()
        self.__sorted_pairs, self.__sorted_indices = self.__sorted_pairs()
        
        self.__upper_bound = None
        self.__lower_bound = -1
        self.__x_opt = np.zeros((super().n, 1))
        
        self.__stop_time = 300
        
        self.__runtime = None
        self.__num_of_nodes = 0
        self.__optimality_gap = None
        self.__stopped = None
        
    @property
    def relative_profits(self):
        return np.copy(self.__relative_profits)
    @property
    def sorted_pairs(self):
        return np.copy(self.__sorted_pairs)
    @property
    def sorted_indices(self):
        return np.copy(self.__sorted_indices)
    @property
    def upper_bound(self):
        return self.__upper_bound
    @property
    def x_opt(self):
        if self.__x_opt is None:
            return None
        reordering_permutation = [np.argwhere(self.sorted_indices == i) for i in range(super().n)]
        return np.copy(self.__x_opt[reordering_permutation])[:, 0, 0]
    @property
    def lower_bound(self):
        return self.__lower_bound
        
    def __relative_profits(self):
        return super().pairs[:, 0]/super().pairs[:, 1]
        
    def __sorted_pairs(self):
        sorted_indeces = np.flip(np.argsort(self.__relative_profits))
        return super().pairs[sorted_indeces, :], sorted_indeces
        
    def __solve_continuous_relaxation(self, fixed_xs:np.array, n:int):
        
        x_star = np.float32((fixed_xs == 1))
        sum_w = x_star.T@self.__sorted_pairs[:, 1]
        h = -1

        while sum_w < self.W and h < n-1:
            h = h + 1
            if fixed_xs[h] == -1:
                sum_w = sum_w + self.__sorted_pairs[h, 1]
                if sum_w > self.W: x_star[h] = (self.W - sum_w + self.__sorted_pairs[h, 1])/self.__sorted_pairs[h, 1]
                else: x_star[h] = 1
                
        lower_bound = np.int64((x_star)).T@self.__sorted_pairs[:, 0]
        upper_bound = x_star.T@self.__sorted_pairs[:, 0]
        sum_w = np.int64((x_star)).T@self.__sorted_pairs[:, 1]
        
        self.__num_of_nodes += 1
        
        return x_star, h, lower_bound, upper_bound, sum_w
        
    def solve(self):
    
        start_time = time.time()
        
        fixed_xs = np.float32(-np.ones((super().n, 1))).reshape(1, -1)

        x_star, frac_var_index, lower_bound, upper_bound, _ = self.__solve_continuous_relaxation(fixed_xs[0], super().n)
        
        self.__upper_bound = upper_bound
        
        node_lower_bounds = np.array([lower_bound])
        node_upper_bounds = np.array([upper_bound])
        open_upper_bounds = None
        if lower_bound == upper_bound:
            self.__lower_bound = upper_bound
            self.__x_opt = np.copy(x_star)
        else: 
            open_upper_bounds = np.array([upper_bound])
            node_fixed_xs = np.copy(fixed_xs)
            node_frac_var_index = np.array([frac_var_index])
        
        self.__stopped = False
        while open_upper_bounds is not None and open_upper_bounds.shape[0] > 0:
        
            if time.time() - start_time >= self.__stop_time:
                self.__stopped = True
                break

            index = np.random.randint(0, open_upper_bounds.shape[0])
            
            fixed_xs_0 = np.copy(node_fixed_xs[index]); fixed_xs_0[node_frac_var_index[index]] = 0
            fixed_xs_1 = np.copy(node_fixed_xs[index]); fixed_xs_1[node_frac_var_index[index]] = 1
            x_star_0, frac_var_index_0, lower_bound_0, upper_bound_0, sum_w_0 = self.__solve_continuous_relaxation(fixed_xs_0, super().n)
            x_star_1, frac_var_index_1, lower_bound_1, upper_bound_1, sum_w_1 = self.__solve_continuous_relaxation(fixed_xs_1, super().n)
            
            open_upper_bounds = np.delete(open_upper_bounds, index)
            node_fixed_xs = np.delete(node_fixed_xs, index, 0)
            node_frac_var_index = np.delete(node_frac_var_index, index)

            if sum_w_0 <= super().W:
                if upper_bound_0 >= self.__lower_bound:
                    node_lower_bounds = np.hstack((node_lower_bounds, lower_bound_0))
                    node_upper_bounds = np.hstack((node_upper_bounds, upper_bound_0))
                    if lower_bound_0 == upper_bound_0:
                        if lower_bound_0 > self.__lower_bound: 
                            self.__x_opt = np.copy(x_star_0)
                            self.__lower_bound = lower_bound_0
                    else: 
                        open_upper_bounds = np.hstack((open_upper_bounds, upper_bound_0))
                        node_fixed_xs = np.vstack((node_fixed_xs, fixed_xs_0))
                        node_frac_var_index = np.hstack((node_frac_var_index, frac_var_index_0))
             
            if sum_w_1 <= super().W:
                if upper_bound_1 >= self.__lower_bound:
                    node_lower_bounds = np.hstack((node_lower_bounds, lower_bound_1))
                    node_upper_bounds = np.hstack((node_upper_bounds, upper_bound_1))
                    if lower_bound_1 == upper_bound_1: 
                        if lower_bound_1 > self.__lower_bound: 
                            self.__x_opt = np.copy(x_star_1)
                            self.__lower_bound = lower_bound_1
                    else: 
                        open_upper_bounds = np.hstack((open_upper_bounds, upper_bound_1))
                        node_fixed_xs = np.vstack((node_fixed_xs, fixed_xs_1))
                        node_frac_var_index = np.hstack((node_frac_var_index, frac_var_index_1))

            keep_indices = (node_upper_bounds >= self.__lower_bound)
            node_upper_bounds = node_upper_bounds[keep_indices]
            node_lower_bounds = node_lower_bounds[keep_indices]

            keep_indices = (open_upper_bounds >= self.__lower_bound)
            open_upper_bounds = open_upper_bounds[keep_indices]
            node_fixed_xs = node_fixed_xs[keep_indices]
            node_frac_var_index = node_frac_var_index[keep_indices]


            if node_upper_bounds.shape[0] > 0:
                self.__upper_bound = np.min(node_upper_bounds)
                
        self.__runtime = time.time() - start_time
        self.__optimality_gap = self.__upper_bound - self.__lower_bound
        
    def stats(self):
        return self.__runtime, self.__num_of_nodes, self.__optimality_gap, self.__stopped, self.__upper_bound, self.__lower_bound

def main():

    path_input, path_output = get_args()
    n, W, pairs = read_problem(path_input)
    pairs = np.array(pairs)
    
    problem = KnapsackProblem(n, W, pairs)
    solver = KnapsackProblemSolver(problem)
    solver.solve()

    # With scipy
    # c = -pairs[:, 0]
    # A = pairs[:, 1]
    # constraints = scipy.optimize.LinearConstraint(A, ub=W)
    # integrality = np.ones_like(c)
    # lb = np.zeros_like(c)
    # ub = np.ones_like(c)
    # bounds = scipy.optimize.Bounds(lb=lb, ub=ub)
    # res = scipy.optimize.milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    # print(res.x)
    
    print(solver.stats())
    
    write_results(path_output, solver.lower_bound, solver.x_opt)

if __name__ == '__main__':
    main()
    


