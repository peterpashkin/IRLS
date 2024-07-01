//
// Created by Peter Pashkin on 11.03.24.
//

#include <Eigen/Core>
#include <Eigen/Dense>
#include "graph_generation/Graph.h"
#include <Eigen/IterativeLinearSolvers>

#ifndef RUNNER_H
#define RUNNER_H



class Runner {
public:
    Runner() = default;
    bool wrong_approx;
    VectorXq linf_minimization(const VectorXq& b, double_custom epsilon, double_custom M, Graph& g);
    VectorXq l1_minimization(const VectorXq& b, double_custom epsilon, double_custom M, Graph& g);
    VectorXq linf_recursive(const VectorXq& b, double_custom epsilon, double_custom M, Graph& g, bool last = true);
    VectorXq l1_recursive(const VectorXq& b, double_custom epsilon, double_custom M, Graph& g, bool last = false);
private:
    static Eigen::ConjugateGradient<Eigen::SparseMatrix<double_custom>, Eigen::Lower|Eigen::Upper> cg;
    static void updateMetaAlpha(VectorXq& alpha, VectorXq& alpha_meta, double_custom exp);
    static void conditionCG(Graph& g);
    static VectorXq compute_phi(Graph& g, const VectorXq& c, const VectorXq& b);
    static VectorXq compute_potential_diff(Graph& g, VectorXq& phi);
    static double_custom linf_invariant(const VectorXq& r_prime, const VectorXq& r, const VectorXq& b, Graph& g, double_custom& energy);
    static VectorXq long_step_alpha(const VectorXq& alpha, double_custom exp);
    static VectorXq compute_x_linf(const VectorXq& r, const VectorXq& b, Graph& g);
    static double_custom lone_invariant(const VectorXq& c_prime, const VectorXq& c, const VectorXq& b, Graph& g, double_custom& energy);
    static VectorXq compute_x_lone(const VectorXq& c, const VectorXq& b, Graph& g);
};

#endif //RUNNER_H
