#pragma once
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif // _OPENMP

// Thread-safe random number generator
class RandomGenerator {
private:
    // Thread-local storage for random generators
    static thread_local std::mt19937 gen;
    
public:
    inline RandomGenerator() {
        std::random_device rd;
        gen = std::mt19937 { rd() + omp_get_thread_num() * 1000 };
    }
    
    static inline constexpr double random_double(double min = 0.0, double max = 1.0) noexcept {
        if (min > max) std::swap(min, max);
        std::uniform_real_distribution<double> dist(min, max);
        return dist(gen);
    }
    
    static inline constexpr int random_int(int min, int max) noexcept {
        if (min > max) std::swap(min, max);
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen);
    }
    
    static inline constexpr bool random_bool() noexcept {
        return random_int(0, 1) ? true : false;
    }
    
    static inline constexpr auto& random_choice(const auto& items) {
        if (items.empty()) throw std::runtime_error("Cannot choose from empty list");
        return items[random_int(0, static_cast<int>(items.size()) - 1)];
    }
    
    template<typename T>
    static inline constexpr std::vector<T> random_sample(const std::vector<T>& population, int k) noexcept {
        if (k <= 0) return {};
        if (k >= static_cast<int>(population.size())) return population;
        
        std::vector<T> result;
        std::vector<int> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        for (int i = 0; i < k; ++i) {
            int idx = random_int(i, static_cast<int>(indices.size()) - 1);
            result.push_back(population[indices[idx]]);
            std::swap(indices[i], indices[idx]);
        }
        
        return result;
    }
};

