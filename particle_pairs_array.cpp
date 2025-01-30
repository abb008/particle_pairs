#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <thread>
#include <future>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <immintrin.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>

const size_t MAX_PARTICLES = 150000;

struct Particle {
    double x, y, z;
};

class ThreadPool {
public:
    ThreadPool(size_t numThreads);
    ~ThreadPool();
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
        workers.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    this->condition.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    if (this->stop && this->tasks.empty()) return;
                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread &worker : workers) worker.join();
}

template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
-> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...)
    );
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

struct KDTreeArray {
    Particle particles[MAX_PARTICLES];
    int indices[MAX_PARTICLES];
    int dimension;
    size_t numParticles;

    KDTreeArray(const Particle* pts, size_t numPts, int dim)
        : dimension(dim), numParticles(numPts) {
        std::copy(pts, pts + numPts, particles);
        std::iota(indices, indices + numPts, 0);
        int parallelDepth = std::log2(std::thread::hardware_concurrency());
        buildTree(0, numParticles, 0, parallelDepth);
    }

    void buildTree(int start, int end, int depth, int parallelDepth) {
        if (start >= end) return;
        int axis = depth % dimension;
        int mid = (start + end) / 2;
        std::nth_element(indices + start, indices + mid, indices + end,
                         [&](int lhs, int rhs) {
            return getCoord(particles[lhs], axis) < getCoord(particles[rhs], axis);
        });
        if (parallelDepth > 0) {
            auto left = std::async(std::launch::async, &KDTreeArray::buildTree,
                                   this, start, mid, depth + 1, parallelDepth - 1);
            auto right = std::async(std::launch::async, &KDTreeArray::buildTree,
                                    this, mid + 1, end, depth + 1, parallelDepth - 1);
            left.get();
            right.get();
        } else {
            buildTree(start, mid, depth + 1, parallelDepth);
            buildTree(mid + 1, end, depth + 1, parallelDepth);
        }
    }

    double getCoord(const Particle& p, int axis) const {
        if (axis == 0) return p.x;
        if (axis == 1) return p.y;
        return p.z;
    }

    double euclideanDistanceSquared(const Particle& p1, const Particle& p2) const {
        __m256d p1_vec = _mm256_set_pd(p1.z, p1.y, p1.x, 0.0);
        __m256d p2_vec = _mm256_set_pd(p2.z, p2.y, p2.x, 0.0);
        __m256d diff = _mm256_sub_pd(p1_vec, p2_vec);
        __m256d sq_diff = _mm256_mul_pd(diff, diff);
        __m256d sum = _mm256_hadd_pd(sq_diff, sq_diff);
        double result[4];
        _mm256_storeu_pd(result, sum);
        return result[0] + result[2];
    }

    void radiusSearch(int node, int start, int end, const Particle& target,
                      double radiusSquared, int depth, std::vector<int>& result) const {
        if (start >= end) return;
        int axis = depth % dimension;
        int mid = (start + end) / 2;
        int idx = indices[mid];
        if (euclideanDistanceSquared(particles[idx], target) <= radiusSquared) {
            result.push_back(idx);
        }
        double diff = getCoord(target, axis) - getCoord(particles[idx], axis);
        if (diff * diff <= radiusSquared) {
            radiusSearch(2 * node + 1, start, mid, target, radiusSquared, depth + 1, result);
            radiusSearch(2 * node + 2, mid + 1, end, target, radiusSquared, depth + 1, result);
        } else if (diff <= 0) {
            radiusSearch(2 * node + 1, start, mid, target, radiusSquared, depth + 1, result);
        } else {
            radiusSearch(2 * node + 2, mid + 1, end, target, radiusSquared, depth + 1, result);
        }
    }
};

void loadParticlesFromFile(const std::string& filename, Particle* particles, size_t& numParticles) {
    std::ifstream file(filename, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read file: " << filename << std::endl;
        return;
    }
    numParticles = 0;
    std::istringstream iss(std::string(buffer.data(), buffer.size()));
    std::string line;
    while (std::getline(iss, line) && numParticles < MAX_PARTICLES) {
        std::istringstream lineStream(line);
        lineStream >> particles[numParticles].x >> particles[numParticles].y >> particles[numParticles].z;
        ++numParticles;
    }
    std::cout << "Loaded " << numParticles << " particles from file." << std::endl;
}

size_t countPairs(const KDTreeArray& tree, double threshold) {
    const double search_radius_squared = threshold * threshold;
    size_t pair_count = 0;
    ThreadPool pool(std::thread::hardware_concurrency());
    auto countPairsInRange = [&](size_t start, size_t end) {
        size_t local_pair_count = 0;
        std::vector<int> ret_matches;
        ret_matches.reserve(tree.numParticles);
        for (size_t i = start; i < end; ++i) {
            ret_matches.clear();
            tree.radiusSearch(0, 0, tree.numParticles,
                              tree.particles[i], search_radius_squared, 0, ret_matches);
            for (int idx : ret_matches) {
                if (idx > static_cast<int>(i)) {
                    ++local_pair_count;
                }
            }
        }
        return local_pair_count;
    };
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t block_size = (tree.numParticles + num_threads - 1) / num_threads;
    std::vector<std::future<size_t>> futures;
    for (size_t t = 0; t < num_threads; ++t) {
        size_t start = t * block_size;
        size_t end = std::min(start + block_size, tree.numParticles);
        futures.emplace_back(pool.enqueue(countPairsInRange, start, end));
    }
    for (auto& f : futures) {
        pair_count += f.get();
    }
    return pair_count;
}

int main() {
    std::string file_path = "positions_large.xyz";
    double threshold = 0.05;
    Particle particles[MAX_PARTICLES];
    size_t numParticles;
    auto start_load = std::chrono::high_resolution_clock::now();
    loadParticlesFromFile(file_path, particles, numParticles);
    auto end_load = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_load = end_load - start_load;
    std::cout << "Time taken to load data: " << duration_load.count() << " seconds" << std::endl;
    auto start_tree = std::chrono::high_resolution_clock::now();
    KDTreeArray tree(particles, numParticles, 3);
    auto end_tree = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_tree = end_tree - start_tree;
    std::cout << "Time taken to build tree: " << duration_tree.count() << " seconds" << std::endl;
    auto start_count = std::chrono::high_resolution_clock::now();
    size_t pair_count = countPairs(tree, threshold);
    auto end_count = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_count = end_count - start_count;
    std::cout << "Number of pairs within " << threshold << "m: " << pair_count << std::endl;
    std::cout << "Time taken to count pairs: " << duration_count.count() << " seconds" << std::endl;
    std::cout << "Total time: " << duration_tree.count() + duration_count.count() << " seconds" << std::endl;
    return 0;
}