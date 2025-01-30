import time
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def load_data(file_path):
    return np.loadtxt(file_path)

def test_kdtree(particles, leafsize):
    t0 = time.time()
    tree = KDTree(particles, leafsize=leafsize, balanced_tree=False, compact_nodes=False, copy_data=True)
    t1 = time.time()
    pair_count = len(tree.query_pairs(r=0.05, p=2, output_type='ndarray'))
    t2 = time.time()
    return pair_count, t1 - t0, t2 - t1, t2 - t0

def optimize_kdtree(particles):
    best_time = float('inf')
    best_leafsize = None
    leafsizes = range(10, 51, 5)
    avg_build_times = []
    avg_query_times = []
    avg_total_times = []

    for leafsize in leafsizes:
        total_build_time = 0
        total_query_time = 0
        total_total_time = 0
        for _ in range(10):
            pair_count, build_time, query_time, total_time = test_kdtree(particles, leafsize)
            total_build_time += build_time
            total_query_time += query_time
            total_total_time += total_time

        avg_build_time = total_build_time / 10
        avg_query_time = total_query_time / 10
        avg_total_time = total_total_time / 10
        avg_build_times.append(avg_build_time)
        avg_query_times.append(avg_query_time)
        avg_total_times.append(avg_total_time)

        print(f"Leafsize: {leafsize}, Pairs: {pair_count}, Avg Build Time: {round(avg_build_time, 5)}, Avg Query Time: {round(avg_query_time, 5)}, Avg Total Time: {round(avg_total_time, 5)}")
        if avg_total_time < best_time:
            best_time = avg_total_time
            best_leafsize = leafsize

    plt.plot(leafsizes, avg_build_times, marker='o', label='Build Time')
    plt.plot(leafsizes, avg_query_times, marker='o', label='Query Time')
    plt.plot(leafsizes, avg_total_times, marker='o', label='Total Time')
    plt.xlabel('Leafsize')
    plt.ylabel('Average Time (seconds)')
    plt.title('KDTree Performance Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_leafsize, best_time

def main():
    file_path = 'positions_large.xyz'
    t0 = time.time()
    particles = load_data(file_path)
    print(f"Data loading time: {round(time.time() - t0, 3)} seconds")

    best_leafsize, best_time = optimize_kdtree(particles)
    print(f"Best leafsize: {best_leafsize}, Best time: {round(best_time, 5)} seconds")

if __name__ == "__main__":
    main()
