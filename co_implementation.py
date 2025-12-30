import numpy as np
import matplotlib.pyplot as plt
from co_classes import CCEvolution, QAP
import os
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


def local_search(perm, flow, distance, max_swaps=20):
    best_perm = perm.copy()
    best_cost = qap_cost(flow, distance, best_perm)
    for _ in range(max_swaps):
        i, j = np.random.choice(len(perm), 2, replace=False)
        new_perm = best_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_cost = qap_cost(flow, distance, new_perm)
        if new_cost < best_cost:
            best_perm = new_perm
            best_cost = new_cost
    return best_perm, best_cost


def global_optimize(best_perms, flow, distance, max_iterations=20):
    best_perm = min(best_perms, key=lambda x: qap_cost(flow, distance, x))
    best_cost = qap_cost(flow, distance, best_perm)
    for _ in range(max_iterations):
        i, j = np.random.choice(len(best_perm), 2, replace=False)
        new_perm = best_perm.copy()
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_cost = qap_cost(flow, distance, new_perm)
        if new_cost < best_cost:
            best_perm = new_perm
            best_cost = new_cost
    return best_perm, best_cost


def build_full_permutation(ind, pop_idx, best_individuals, subsets):
    full_perm = np.zeros(len(flow_matrix), dtype=int)
    offset = 0
    for i, subset in enumerate(subsets):
        perm = ind.value if i == pop_idx else best_individuals[i].value
        for idx, val in enumerate(perm):
            full_perm[subset[idx]] = val + offset
        offset += len(subset)
    return full_perm


def ccea_fitness_creator(flow, distance, subsets, top_individuals):
    def fitness(ind, pop_idx):
        min_cost = float('inf')
        best_perm = None
        for combo in np.ndindex(*[len(top_individuals[i]) for i in range(len(subsets)) if i != pop_idx]):
            others = []
            combo_idx = 0
            for i in range(len(subsets)):
                if i == pop_idx:
                    others.append(ind)
                else:
                    others.append(top_individuals[i][combo[combo_idx]])
                    combo_idx += 1
            full_perm = np.zeros(len(flow_matrix), dtype=int)
            offset = 0
            for i, subset in enumerate(subsets):
                perm = others[i].value
                for idx, val in enumerate(perm):
                    full_perm[subset[idx]] = val + offset
                offset += len(subset)
            cost = qap_cost(flow, distance, full_perm)
            if cost < min_cost:
                min_cost = cost
                best_perm = full_perm
        return min_cost

    return fitness


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def run_ccea(n_populations, pool_size, n_offsprings, seed):
    set_random_seed(seed)
    
    subsets = np.array_split(np.arange(12), n_populations)
    init_params_list = [{'n_facilities': len(subset)} for subset in subsets]

    ccea = CCEvolution(
        pool_size=pool_size,
        individual_class=QAP,
        n_offsprings=n_offsprings,
        pair_params={'alpha': 0.7},
        mutate_params={'rate': 6},
        init_params_list=init_params_list
    )

    n_epochs = 1000
    hist_ccea = []
    best_cost_ever = float('inf')
    best_perm_ever = None
    generations_to_converge = n_epochs

    for i in range(n_epochs):
        top_individuals = []
        for pop_idx, pool in enumerate(ccea.pools):
            pool.fitness = lambda ind, idx=pop_idx: qap_cost(flow_matrix, distance_matrix, build_full_permutation(ind, idx,
                                                                                                                  [ccea.pools[j].individuals[0] for j in range(len(subsets))],
                                                                                                                  subsets))
            sorted_individuals = sorted(pool.individuals, key=lambda x: pool.fitness(x, pop_idx))
            top_individuals.append(sorted_individuals[:1])

        for pop_idx, pool in enumerate(ccea.pools):
            pool.fitness = lambda ind, idx=pop_idx: ccea_fitness_creator(flow_matrix, distance_matrix, subsets,
                                                                         top_individuals)(ind, idx)

        best_costs = []
        best_perms = []
        for pop_idx, pool in enumerate(ccea.pools):
            individuals = sorted(pool.individuals, key=lambda x: pool.fitness(x, pop_idx))
            best_ind = individuals[0]
            full_perm = build_full_permutation(best_ind, pop_idx, [top_individuals[j][0] for j in range(len(subsets))],
                                               subsets)
            full_perm, cost = local_search(full_perm, flow_matrix, distance_matrix)
            best_costs.append(cost)
            best_perms.append(full_perm)

        if i % 50 == 0 and i > 0:
            best_perm, best_cost = global_optimize(best_perms, flow_matrix, distance_matrix)
            if best_cost < min(best_costs):
                best_costs.append(best_cost)
                best_perms.append(best_perm)

        min_cost = min(best_costs)
        if min_cost < best_cost_ever:
            best_cost_ever = min_cost
            best_perm_ever = best_perms[np.argmin(best_costs)]
            if best_cost_ever <= ACCEPTABLE_COST:
                generations_to_converge = i + 1
        
        hist_ccea.append(min_cost)
        
        if min_cost <= OPTIMAL_COST:
            generations_to_converge = i + 1
            break

        ccea.step()

    return best_cost_ever, generations_to_converge, hist_ccea, best_perm_ever


def run_multiple_trials(n_trials, n_populations, pool_size, n_offsprings):
    results = []
    for trial in range(n_trials):
        seed = int(time.time() * 1000) % (2**32 - 1)
        best_cost, generations, history, best_perm = run_ccea(n_populations, pool_size, n_offsprings, seed)
        results.append({
            'trial': trial + 1,
            'seed': seed,
            'best_cost': best_cost,
            'generations': generations,
            'history': history,
            'best_perm': best_perm
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
    plt.title('Convergence of CCEA')
    plt.xlabel('Generation')
    plt.ylabel('Cost')
    plt.yscale('log')
    plt.savefig(filename)
    plt.close()


def run_ccea_for_population(n_populations):
    print(f"Running CCEA with {n_populations} populations")
    pool_size = 200 // n_populations
    n_offsprings = 60 // n_populations
    results = run_multiple_trials(n_trials=10, n_populations=n_populations, pool_size=pool_size, n_offsprings=n_offsprings)
    summary = analyze_results(results)
    
    with open(f"results/ccea_summary_pop{n_populations}.txt", "w") as f:
        f.write(f"CCEA Summary (Number of Populations: {n_populations})\n")
        f.write(f"Best Overall Cost: {summary['best_overall']}\n")
        f.write(f"Worst Overall Cost: {summary['worst_overall']}\n")
        f.write(f"Average Cost: {summary['avg_cost']:.2f} ± {summary['std_cost']:.2f}\n")
        f.write(f"Average Generations: {summary['avg_generations']:.2f} ± {summary['std_generations']:.2f}\n")
        f.write(f"Total Trials: {summary['n_trials']}\n\n")
        
        for r in results:
            f.write(f"Trial {r['trial']}, Seed: {r['seed']}, Best Cost: {r['best_cost']}, Generations: {r['generations']}\n")
            f.write(f"Best Permutation: {r['best_perm']}\n\n")
    
    plot_convergence(results, f"results/ccea_convergence_pop{n_populations}.png")
    
    return n_populations, summary

def main():
    os.makedirs("results", exist_ok=True)
    
    population_sizes = [2, 4, 6]
    
    with multiprocessing.Pool() as pool:
        results = pool.map(run_ccea_for_population, population_sizes)
    
    for n_populations, summary in results:
        print(f"CCEA with {n_populations} populations:")
        print(f"  Best Overall Cost: {summary['best_overall']}")
        print(f"  Average Cost: {summary['avg_cost']:.2f} ± {summary['std_cost']:.2f}")
        print(f"  Average Generations: {summary['avg_generations']:.2f} ± {summary['std_generations']:.2f}")
        print()
    
    print("CCEA execution completed. Results saved in the results folder.")

if __name__ == '__main__':
    main()