#include "cobyla.hpp"
#include <iostream>

// COBYLA implementation for parameter optimization
Cobyla::Result Cobyla::minimize(const std::vector<double>& x0) const {
    Result result;
    result.x = x0;
    result.nfev = 0;
    result.success = false;

    if (!objective_) {
        std::cerr << "Objective function not set!" << std::endl;
        return result;
    }

    // Simplified COBYLA implementation
    std::vector<double> x = x0;
    double current_rho = rhobeg_;
    double best_fval = objective_(x);
    result.nfev++;

    int iterations = 0;
    while (current_rho > rhoend_ && iterations < maxfun_ / 10) {
        bool improved = false;

        // Generate trial points
        for (int i = 0; i < 2 * n_; ++i) {
            std::vector<double> trial = x;

            // Perturb parameters
            for (size_t j = 0; j < trial.size(); ++j) {
                double perturbation = ((j % 2 == 0) ? 1 : -1) * current_rho;
                trial[j] += perturbation;

                // Keep angles in [0, 2π]
                if (trial[j] < 0) trial[j] += 2 * M_PI;
                if (trial[j] > 2 * M_PI) trial[j] -= 2 * M_PI;
            }

            double trial_fval = objective_(trial);
            result.nfev++;

            if (trial_fval < best_fval) {
                best_fval = trial_fval;
                x = trial;
                improved = true;
            }

            if (result.nfev >= maxfun_) break;
        }

        if (!improved) {
            current_rho *= 0.5;
        }

        iterations++;
        if (result.nfev >= maxfun_) break;
    }

    result.x = x;
    result.fun = best_fval;
    result.success = true;

    return result;
}
