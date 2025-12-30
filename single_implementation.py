
import numpy as np
import matplotlib.pyplot as plt
from single_classes import Evolution, QAP
import os
from concurrent.futures import ProcessPoolExecutor
import random
import time
import multiprocessing

OPTIMAL_COST = 9552
ACCEPTABLE_COST = 11000

# QAPLIB chr12a flow matrix
flow_matrix = np.array([
    [0, 90, 10, 23, 43, 0, 0, 0, 0, 0, 0, 0],
    [90, 0, 0, 0, 0, 88, 0, 0, 0, 0, 0, 0],
    [10, 0, 0, 0, 0, 0, 26, 16, 0, 0, 0, 0],
    [23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 88, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 16, 0, 0, 0, 0, 0, 0, 96, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 29, 0],
    [0, 0, 0, 0, 0, 0, 0, 96, 0, 0, 0, 37],
    [0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 0, 0]
])

# QAPLIB chr12a distance matrix
distance_matrix = np.array([
    [0, 36, 54, 26, 59, 72, 9, 34, 79, 17, 46, 95],
    [36, 0, 73, 35, 90, 58, 30, 78, 35, 44, 79, 36],
    [54, 73, 0, 21, 10, 97, 58, 66, 69, 61, 54, 63],
    [26, 35, 21, 0, 93, 12, 46, 40, 37, 48, 68, 85],
    [59, 90, 10, 93, 0, 64, 5, 29, 76, 16, 5, 76],
    [72, 58, 97, 12, 64, 0, 96, 55, 38, 54, 0, 34],
    [9, 30, 58, 46, 5, 96, 0, 83, 35, 11, 56, 37],
    [34, 78, 66, 40, 29, 55, 83, 0, 44, 12, 15, 80],
    [79, 35, 69, 37, 76, 38, 35, 44, 0, 64, 39, 33],
    [17, 44, 61, 48, 16, 54, 11, 12, 64, 0, 70, 86],
    [46, 79, 54, 68, 5, 0, 56, 15, 39, 70, 0, 18],
    [95, 36, 63, 85, 76, 34, 37, 80, 33, 86, 18, 0]
])

def qap_fitness_creator(flow, distance):
    def fitness(qap):
        cost = 0
        perm = qap.value
        for i in range(len(perm)):
            for j in range(len(perm)):
                cost += flow[i, j] * distance[perm[i], perm[j]]
        return cost
    return fitness

def qap_cost(flow, distance, perm):
    if not (len(perm) == flow.shape[0] == distance.shape[0]):
        print(f"Error: Invalid dimensions: perm={len(perm)}, flow={flow.shape}, distance={distance.shape}")
        return float('inf')
    cost = 0
    for i in range(len(perm)):
        for j in range(len(perm)):
            cost += flow[i, j] * distance[perm[i], perm[j]]
    if cost < 9552:  # Optimal cost for chr12a
        print(f"Warning: Cost {cost} below optimal 9552 for perm={perm}")
    return cost

def run_and_return(run):
    seed = int(time.time() * 1000) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"Running Single EA - Run {run} with seed {seed}")
    
    fitness = qap_fitness_creator(flow_matrix, distance_matrix)
    evo = Evolution(
        pool_size=100,
        fitness=fitness,
        individual_class=QAP,
        n_offsprings=30,
        pair_params={'alpha': 0.1},  # More exploration
        mutate_params={'rate': 6},   # More mutation
        init_params={'n_facilities': 12}
    )

    # Run EA
    n_epochs = 2000
    best_cost_ever = float('inf')
    generations_to_converge = n_epochs

    for i in range(n_epochs):
        best_ind = evo.pool.individuals[0]
        cost = qap_cost(flow_matrix, distance_matrix, best_ind.value)
        if cost < best_cost_ever:
            best_cost_ever = cost
            if best_cost_ever <= ACCEPTABLE_COST:
                generations_to_converge = i + 1
        if cost <= OPTIMAL_COST:
            generations_to_converge = i + 1
            break
        evo.step()

    line = f"Run {run}, Best Cost: {best_cost_ever}, Generations: {generations_to_converge}"
    return line, best_cost_ever, generations_to_converge

def run_single_ea(seed, pool_size=100, n_offsprings=30):
    random.seed(seed)
    np.random.seed(seed)
    
    fitness = qap_fitness_creator(flow_matrix, distance_matrix)
    evo = Evolution(
        pool_size=pool_size,
        fitness=fitness,
        individual_class=QAP,
        n_offsprings=n_offsprings,
        pair_params={'alpha': 0.1},
        mutate_params={'rate': 6},
        init_params={'n_facilities': 12}
    )

    n_epochs = 2000
    best_cost_ever = float('inf')
    generations_to_converge = n_epochs
    history = []

    for i in range(n_epochs):
        best_ind = evo.pool.individuals[0]
        cost = qap_cost(flow_matrix, distance_matrix, best_ind.value)
        history.append(cost)
        if cost < best_cost_ever:
            best_cost_ever = cost
            if best_cost_ever <= ACCEPTABLE_COST:
                generations_to_converge = i + 1
        if cost <= OPTIMAL_COST:
            generations_to_converge = i + 1
            break
        evo.step()

    return best_cost_ever, generations_to_converge, history

def run_multiple_trials(n_trials=10, pool_size=100, n_offsprings=30):
    results = []
    for trial in range(n_trials):
        seed = int(time.time() * 1000) % (2**32 - 1)
        best_cost, generations, history = run_single_ea(seed, pool_size, n_offsprings)
        results.append({
            'trial': trial + 1,
            'seed': seed,
            'best_cost': best_cost,
            'generations': generations,
            'history': history
        })
    return results

def analyze_results(results):
    best_costs = [r['best_cost'] for r in results]
    generations = [r['generations'] for r in results]
    
    summary = {
        'best_overall': min(best_costs),
        'worst_overall': max(best_costs),
        'avg_cost': np.mean(best_costs),
        'std_cost': np.std(best_costs),
        'avg_generations': np.mean(generations),
        'std_generations': np.std(generations),
        'n_trials': len(results)
    }
    
    return summary

def plot_convergence(results, filename):
    plt.figure(figsize=(10, 6))
    for r in results:
        plt.plot(r['history'], alpha=0.3)
    plt.title('Convergence of Single EA')
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.yscale('log')
    plt.savefig(filename)
    plt.close()

def run_single_ea_for_pool_size(pool_size):
    print(f"Running Single EA with pool size {pool_size}")
    n_offsprings = int(pool_size * 0.3)
    results = run_multiple_trials(n_trials=10, pool_size=pool_size, n_offsprings=n_offsprings)
    summary = analyze_results(results)
    
    with open(f"results/single_ea_summary_pool{pool_size}.txt", "w") as f:
        f.write(f"Single EA Summary (Pool Size: {pool_size})\n")
        f.write(f"Best Overall Cost: {summary['best_overall']}\n")
        f.write(f"Worst Overall Cost: {summary['worst_overall']}\n")
        f.write(f"Average Cost: {summary['avg_cost']:.2f} ± {summary['std_cost']:.2f}\n")
        f.write(f"Average Generations: {summary['avg_generations']:.2f} ± {summary['std_generations']:.2f}\n")
        f.write(f"Total Trials: {summary['n_trials']}\n\n")
        
        for r in results:
            f.write(f"Trial {r['trial']}, Seed: {r['seed']}, Best Cost: {r['best_cost']}, Generations: {r['generations']}\n")
    
    plot_convergence(results, f"results/single_ea_convergence_pool{pool_size}.png")
    
    return pool_size, summary

def main():
    os.makedirs("results", exist_ok=True)
    
    pool_sizes = [100, 200, 300]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(run_single_ea_for_pool_size, pool_sizes)
    
    for pool_size, summary in results:
        print(f"Single EA with pool size {pool_size}:")
        print(f"  Best Overall Cost: {summary['best_overall']}")
        print(f"  Average Cost: {summary['avg_cost']:.2f} ± {summary['std_cost']:.2f}")
        print(f"  Average Generations: {summary['avg_generations']:.2f} ± {summary['std_generations']:.2f}")
        print()
    
    print("Single EA execution completed. Results saved in the results folder.")

if __name__ == '__main__':
    main()
