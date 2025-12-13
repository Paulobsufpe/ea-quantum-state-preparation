#pragma once
#include <functional>
#include <vector>
#include <math.h>

// COBYLA implementation for parameter optimization

class Cobyla {
private:
    std::function<double(const std::vector<double>&)> objective_;
    int n_;
    // TODO: fix unused
    [[maybe_unused]] int m_;
    double rhobeg_;
    double rhoend_;
    int maxfun_;
    
public:
    inline Cobyla(int n, int m, double rhobeg = 0.5, double rhoend = 1e-6, int maxfun = 1000)
        : n_(n), m_(m), rhobeg_(rhobeg), rhoend_(rhoend), maxfun_(maxfun) {}
    
    inline void set_objective(std::function<double(const std::vector<double>&)> objective) {
        objective_ = objective;
    }
    
    struct Result {
        std::vector<double> x;
        double fun;
        bool success;
        int nfev;
    };
    
    Result minimize(const std::vector<double>& x0) const;
};
