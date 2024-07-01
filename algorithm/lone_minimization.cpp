#include "../graph_generation/Graph.h"
#include "../Runner.h"

VectorXq Runner::l1_minimization(const VectorXq& b, double_custom epsilon, double_custom M, Graph& g) {
    wrong_approx = true;
    int n = g.n;
    int m = g.m;
    conditionCG(g);

    VectorXq c = VectorXq::Ones(m) / m;

    int t_prime = 0;
    VectorXq s_t_prime = VectorXq::Zero(m);
    VectorXq phi_t_prime = VectorXq::Zero(n);

    VectorXq phi;
    VectorXq alpha_meta_significant = VectorXq::Zero(m);


    while (c.lpNorm<1>() <= 1 + 1/(std::pow((1 + epsilon), 2) - 1)) {
        phi = compute_phi(g, c, b);

        VectorXq scaled_phi = phi / (b.transpose() * phi);
        VectorXq tmp = compute_potential_diff(g, scaled_phi);

        if(tmp.lpNorm<Eigen::Infinity>() <= std::pow(m, 1.0 / 3) / M) {
            t_prime += 1;
            s_t_prime += tmp.cwiseAbs();
            phi_t_prime += scaled_phi;
        }

        if(s_t_prime.lpNorm<Eigen::Infinity>() / t_prime <= (1/((1.0-epsilon)*M))) {
            return phi_t_prime / t_prime;
        }

        VectorXq alpha_t = VectorXq::Ones(m);
        bool changed = false;
        for (int i = 0; i < m; ++i) {
            if(abs(tmp(i)) > 1/((1-epsilon) * M)) {
                alpha_t(i) = std::pow(tmp[i], 2) * pow(M, 2);
                changed = true;
            }
        }

        if(!changed) {
            return phi;
        }

        VectorXq old_c = c;
        double_custom energy = b.transpose() * compute_phi(g, c, b);
#ifdef LONG_STEP

        VectorXq alpha_significant = alpha_t - VectorXq::Ones(m);
        double_custom base = 2;
        double_custom exp = 1;
        while(c.lpNorm<1>() <= 1 + 1/(std::pow((1 + epsilon), 2) - 1)) {
            VectorXq c_prime = old_c.cwiseProduct(long_step_alpha(alpha_significant, exp));
            double_custom val = lone_invariant(c_prime, c, b, g, energy);
            if(val < 1.0/(M*M)) {
                break;
            }
            c = c_prime;
            exp *= base;
        }

        updateMetaAlpha(alpha_significant, alpha_meta_significant, exp);

        exp = 1;
        old_c = c;
        energy = b.transpose() * compute_phi(g, c, b);
        while(c.lpNorm<1>() <= 1 + 1/(std::pow((1 + epsilon), 2) - 1)) {
            VectorXq c_prime = old_c.cwiseProduct(long_step_alpha(alpha_meta_significant, exp));
            double_custom val = lone_invariant(c_prime, c, b, g, energy);
            if(val < 1.0/(M*M)) {
                break;
            }
            c = c_prime;
            exp *= base;
        }

#else
        c = c.cwiseProduct(alpha_t);
#endif
    }
    wrong_approx = false;
    return compute_x_lone(c, b, g);
}