#include "../graph_generation/Graph.h"
#include "../Runner.h"

VectorXq Runner::linf_recursive(const VectorXq& b, double_custom epsilon, double_custom M, Graph& g, bool last) {
    using namespace Eigen;
    int m = g.m;
    wrong_approx = false;
    conditionCG(g);

    VectorXq r = VectorXq::Ones(m) / m;

    // -------------------------------RECURSION--------------------------------

    if(epsilon < 0.5) {
        VectorXq result = linf_recursive(b, 1.3 * epsilon, (1 - 0.5*epsilon) * M, g, false);
        if(!wrong_approx) {
            return result;
        }
        r = result;
    }

    //--------------------------------------------------------------------------

    int t_prime = 0;
    VectorXq s_t_prime = VectorXq::Zero(m);
    VectorXq phi;

    VectorXq alpha_meta_significant = VectorXq::Zero(m);

    while (r.lpNorm<1>() <= 1.0/epsilon) {
        VectorXq x = compute_x_linf(r, b, g);

        if (x.lpNorm<Infinity>() <= std::pow(m, 1.0 / 3) * M) {
            t_prime += 1;
            s_t_prime += x;
        }


        if (s_t_prime.lpNorm<Infinity>() / t_prime <= (1.0 + epsilon) * M) {
            wrong_approx = false;
            return s_t_prime / t_prime;
        }

        VectorXq alpha_t = VectorXq::Ones(m);
        bool changed = false;
        for (int i = 0; i < m; ++i) {
            double_custom x_t_i = x(i);
            if (std::abs(x_t_i) >= (1.0 + epsilon) * M) {
                alpha_t(i) = std::pow(x_t_i, 2) / std::pow(M, 2);
                changed = true;
            }
        }

        if (!changed) {
            wrong_approx = false;
            return x;
        }

        double_custom energy = b.transpose() * compute_phi(g, r.array().inverse(), b);
        VectorXq old_r = r;

#ifdef LONG_STEP

        VectorXq alpha_significant = alpha_t - VectorXq::Ones(m);
        const double_custom base = 2;
        double_custom exp = 1;
        while(r.lpNorm<1>() <= 1.0/epsilon) {
            VectorXq r_prime = old_r.cwiseProduct(VectorXq::Ones(m) + (exp * alpha_significant));
            double_custom val = linf_invariant(r_prime, r, b, g, energy);
            if(val < M*M) {
                break;
            }
            r = r_prime;
            exp *= base;
        }

        updateMetaAlpha(alpha_significant,alpha_meta_significant, exp);

        exp = 1;
        old_r = r;
        energy = b.transpose() * compute_phi(g, r.array().inverse(), b);
        while(r.lpNorm<1>() <= 1.0/epsilon) {
            VectorXq r_prime = old_r.cwiseProduct(long_step_alpha(alpha_meta_significant, exp));
            double_custom val = linf_invariant(r_prime, r, b, g, energy);
            if(val < M*M) {
                break;
            }
            r = r_prime;
            exp *= base;
        }

#else
        r = r.cwiseProduct(alpha_t);
#endif

    }

    wrong_approx = true;
    return r / r.lpNorm<1>();
}