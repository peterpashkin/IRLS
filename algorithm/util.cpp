#include "../graph_generation/Graph.h"
#include "../Runner.h"

Eigen::ConjugateGradient<Eigen::SparseMatrix<double_custom>, Eigen::Lower|Eigen::Upper> Runner::cg;

void Runner::updateMetaAlpha(VectorXq& alpha, VectorXq& alpha_meta, double_custom exp) {
    const double_custom delta = 0.1;
    alpha_meta *= (1 - delta);
    alpha_meta += delta * alpha * exp;
}

void Runner::conditionCG(Graph& g) {
    int n = g.n;
    int m = g.m;
    Eigen::SparseMatrix<double_custom> L(n, n);
    L.reserve(n + 2*m);
    for(int i=0; i<n; i++) {
        auto& neighbours = g.adj_list[i];
        for(auto neighbour : neighbours) {
            L.coeffRef(i,i) += 1;
            L.coeffRef(i, neighbour.first) -= 1;
        }
    }
    cg.analyzePattern(L);
}

VectorXq Runner::compute_phi(Graph& g, const VectorXq& c, const VectorXq& b) {
    int n = g.n;
    int m = g.m;
    Eigen::SparseMatrix<double_custom> L(n, n);
    L.reserve(n + 2*m);

    for(int i=0; i<n; i++) {
        auto& neighbours = g.adj_list[i];
        for(auto neighbour : neighbours) {
            L.coeffRef(i,i) += c(neighbour.second);
            L.coeffRef(i, neighbour.first) -= c(neighbour.second);
        }
    }

    VectorXq phi = cg.factorize(L).solve(b);
    return phi;
}


VectorXq Runner::compute_potential_diff(Graph& g, VectorXq& phi) {
    VectorXq res = VectorXq::Zero(g.m);
    for (int i = 0; i < g.n; ++i) {
        auto& neighbours = g.adj_list[i];
        for(auto neighbour : neighbours) {
            auto edge = g.edges[neighbour.second];
            if(edge.first == i)
                res[neighbour.second] = phi(i) - phi(neighbour.first);
            else
                res[neighbour.second] = -phi(i) + phi(neighbour.first);
        }
    }
    return res;
}

double_custom Runner::linf_invariant(const VectorXq& r_prime, const VectorXq& r, const VectorXq& b, Graph& g, double_custom& energy) {
    double_custom energy_prime = b.transpose() * compute_phi(g, r_prime.array().inverse(), b);
    double_custom dif = (r_prime - r).lpNorm<1>();
    double_custom val = (energy_prime - energy) / dif;
    energy = energy_prime;
    return val;
}

VectorXq Runner::long_step_alpha(const VectorXq &alpha, double_custom exp) {
    return VectorXq::Ones(alpha.rows()) + (exp * alpha);
}

VectorXq Runner::compute_x_linf(const VectorXq& r, const VectorXq& b, Graph& g) {
    VectorXq c = r.array().inverse();
    VectorXq phi = compute_phi(g, c, b);
    VectorXq potent_diff = compute_potential_diff(g, phi);
    return c.cwiseProduct(potent_diff);
}

double_custom Runner::lone_invariant(const VectorXq& c_prime, const VectorXq& c, const VectorXq& b, Graph& g, double_custom& energy) {
    double_custom energy_prime = b.transpose() * compute_phi(g, c_prime, b);
    double_custom dif = (c_prime - c).lpNorm<1>();
    double_custom val = (1.0/energy_prime - 1.0/energy) / dif;
    energy = energy_prime;
    return val;
}

VectorXq Runner::compute_x_lone(const VectorXq& c, const VectorXq& b, Graph& g) {
    VectorXq phi = compute_phi(g, c, b);
    VectorXq potent_diff = compute_potential_diff(g, phi);
    return c.cwiseProduct(potent_diff);
}