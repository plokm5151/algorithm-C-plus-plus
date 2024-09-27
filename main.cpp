#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <nlopt.hpp>
#include <algorithm>

using namespace std;
using namespace Eigen;

// Struct to hold data required for optimization
struct OptimizationData {
    VectorXd col; // Vector containing average Hellinger distances
};

// Function declarations
MatrixXd computeScoreMatrix(const MatrixXd& DM);
MatrixXd computeHesitancyDegrees(const MatrixXd& DM);
double computeHellingerDistance(const VectorXd& ivifv1, const VectorXd& ivifv2);
vector<int> rankScores(const VectorXd& scores);
double objectiveFunction(const vector<double>& x, vector<double>& grad, void* data);
double equalityConstraint(const vector<double>& x, vector<double>& grad, void* data);

int main() {
    // Initialize the decision matrix DM (3 alternatives, 3 criteria, each with 4 IVIFV components)
    MatrixXd DM(3, 12);
    DM << 0.95, 1.0, 0.0, 0.0,   0.0, 0.0, 0.95, 1.0,    0.0, 0.0, 0.95, 1.0,
          0.90, 1.0, 0.0, 0.0,   0.0, 0.0, 0.90, 1.0,    0.0, 0.0, 0.90, 1.0,
          0.85, 1.0, 0.0, 0.0,   0.0, 0.0, 0.85, 1.0,    0.0, 0.0, 0.85, 1.0;

    // Step 1: Calculate score matrix SM and hesitancy degrees H
    MatrixXd SM = computeScoreMatrix(DM);
    MatrixXd H = computeHesitancyDegrees(DM);

    // Step 2: Set up the optimization problem
    int numVariables = 9; // Number of optimization variables
    VectorXd lb = VectorXd::Zero(numVariables); // Lower bounds
    VectorXd ub = VectorXd::Ones(numVariables); // Upper bounds

    // Extract IVIFVs from DM for Hellinger distance calculations
    vector<VectorXd> ivifvs(9); // 3 alternatives x 3 criteria
    for (int i = 0; i < 3; ++i) { // Alternatives
        for (int j = 0; j < 3; ++j) { // Criteria
            ivifvs[i * 3 + j] = DM.row(i).segment(j * 4, 4);
        }
    }

    // Compute average Hellinger distances for each criterion
    VectorXd col(3);
    for (int c = 0; c < 3; ++c) { // For each criterion
        double dis1_2 = computeHellingerDistance(ivifvs[0 * 3 + c], ivifvs[1 * 3 + c]);
        double dis1_3 = computeHellingerDistance(ivifvs[0 * 3 + c], ivifvs[2 * 3 + c]);
        double dis2_3 = computeHellingerDistance(ivifvs[1 * 3 + c], ivifvs[2 * 3 + c]);

        double avgDistance = (dis1_2 + dis1_3 + dis2_3) / 3.0;
        avgDistance = round(avgDistance * 1000.0) / 1000.0;
        col(c) = avgDistance;

        cout << "col" << c + 1 << " = " << col(c) << endl;
    }

    // Initial guess for optimization variables
    vector<double> x0(numVariables, 0.5);

    // Optimization data
    OptimizationData optData;
    optData.col = col;

    // Step 3: Set up the optimizer
    nlopt::opt localOpt(nlopt::LD_MMA, numVariables); // Local optimizer
    nlopt::opt opt(nlopt::AUGLAG, numVariables);      // Global optimizer with equality constraints

    opt.set_local_optimizer(localOpt);

    vector<double> lbVec(lb.data(), lb.data() + lb.size());
    vector<double> ubVec(ub.data(), ub.data() + ub.size());
    opt.set_lower_bounds(lbVec);
    opt.set_upper_bounds(ubVec);

    opt.set_min_objective(objectiveFunction, &optData);
    opt.add_equality_constraint(equalityConstraint, NULL, 1e-8);

    // Set optimization parameters
    opt.set_xtol_rel(1e-4);
    opt.set_maxeval(500);

    // Run the optimization
    vector<double> x = x0;
    double minf;
    try {
        nlopt::result result = opt.optimize(x, minf);
    } catch (std::exception& e) {
        cout << "Optimization failed: " << e.what() << endl;
        return EXIT_FAILURE;
    }

    // Extract optimized weights
    VectorXd W(numVariables);
    cout << "Optimized weights W = [";
    for (int i = 0; i < numVariables; ++i) {
        W(i) = round(x[i] * 1000.0) / 1000.0;
        cout << W(i);
        if (i != numVariables - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;

    // Step 4: Calculate weighted scores for each alternative
    int numAlternatives = SM.rows();
    int numCriteria = SM.cols();

    VectorXd weightedScores(numAlternatives);
    for (int i = 0; i < numAlternatives; ++i) {
        double score = 0.0;
        for (int j = 0; j < numCriteria; ++j) {
            score += W(j) * SM(i, j);
        }
        weightedScores(i) = round(score * 1000.0) / 1000.0;
        cout << "Weighted score for alternative " << i + 1 << ": " << weightedScores(i) << endl;
    }

    // Rank the alternatives based on weighted scores
    vector<int> rankings = rankScores(weightedScores);

    // Output rankings
    cout << "Rankings: [";
    for (size_t i = 0; i < rankings.size(); ++i) {
        cout << rankings[i];
        if (i != rankings.size() - 1) {
            cout << ", ";
        }
    }
    cout << "]" << endl;

    return EXIT_SUCCESS;
}

// Function to compute the score matrix SM from decision matrix DM
MatrixXd computeScoreMatrix(const MatrixXd& DM) {
    int numAlternatives = DM.rows();
    int numCriteria = DM.cols() / 4; // Each criterion has 4 components

    MatrixXd SM(numAlternatives, numCriteria);

    auto computeS = [](double a, double b, double c, double d) {
        double value = 0.5 * (a - c + b - d)
            + (sin(0.5 * M_PI * a) + sin(0.5 * M_PI * b)) / (1 + sin(0.5 * M_PI * c) + sin(0.5 * M_PI * d))
            + (cos(0.5 * M_PI * c) + cos(0.5 * M_PI * d)) / (1 + cos(0.5 * M_PI * a) + cos(0.5 * M_PI * b))
            + 1;
        return round(value * 1000) / 1000.0;
    };

    for (int i = 0; i < numAlternatives; ++i) {
        for (int j = 0; j < numCriteria; ++j) {
            double a = DM(i, j * 4 + 0);
            double b = DM(i, j * 4 + 1);
            double c = DM(i, j * 4 + 2);
            double d = DM(i, j * 4 + 3);
            SM(i, j) = computeS(a, b, c, d);
        }
    }

    return SM;
}

// Function to compute the hesitancy degrees H from decision matrix DM
MatrixXd computeHesitancyDegrees(const MatrixXd& DM) {
    int numAlternatives = DM.rows();
    int numCriteria = DM.cols() / 4;

    MatrixXd H(numAlternatives, numCriteria * 2);

    for (int i = 0; i < numAlternatives; ++i) {
        for (int j = 0; j < numCriteria; ++j) {
            double a = DM(i, j * 4 + 0);
            double b = DM(i, j * 4 + 1);
            double c = DM(i, j * 4 + 2);
            double d = DM(i, j * 4 + 3);

            double h1 = 1 - b - d;
            double h2 = 1 - a - c;

            H(i, j * 2) = h1;
            H(i, j * 2 + 1) = h2;
        }
    }

    return H;
}

// Function to compute the Hellinger distance between two IVIFVs
double computeHellingerDistance(const VectorXd& ivifv1, const VectorXd& ivifv2) {
    double a = ivifv1(0), b = ivifv1(1), c = ivifv1(2), d = ivifv1(3);
    double e = ivifv2(0), f = ivifv2(1), g = ivifv2(2), h = ivifv2(3);

    double s1 = sqrt(a) - sqrt(e);
    double s2 = sqrt(b) - sqrt(f);
    double s3 = sqrt(c) - sqrt(g);
    double s4 = sqrt(d) - sqrt(h);
    double s5 = sqrt(1 - b - d) - sqrt(1 - f - h);
    double s6 = sqrt(1 - a - c) - sqrt(1 - e - g);

    double distance = 0.5 * sqrt(s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4 + s5 * s5 + s6 * s6);
    return distance;
}

// Function to rank the scores in descending order
vector<int> rankScores(const VectorXd& scores) {
    vector<pair<double, int>> scoreIndexPairs;
    int n = scores.size();
    for (int i = 0; i < n; ++i) {
        scoreIndexPairs.push_back(make_pair(scores(i), i + 1)); // Index starts from 1
    }

    sort(scoreIndexPairs.begin(), scoreIndexPairs.end(), greater<pair<double, int>>());

    vector<int> ranks(n);
    int currentRank = 1;
    for (size_t i = 0; i < scoreIndexPairs.size(); ++i) {
        if (i > 0 && scoreIndexPairs[i].first < scoreIndexPairs[i - 1].first) {
            currentRank = i + 1;
        }
        int originalIndex = scoreIndexPairs[i].second - 1; // Adjust to 0-based index
        ranks[originalIndex] = currentRank;
    }

    return ranks;
}

// Objective function for the optimization problem
double objectiveFunction(const vector<double>& x, vector<double>& grad, void* data) {
    OptimizationData* optData = reinterpret_cast<OptimizationData*>(data);
    VectorXd xv = VectorXd::Map(x.data(), x.size());

    double obj = -(optData->col(0) * xv(0) + optData->col(1) * xv(1) + optData->col(2) * xv(2))
                 + xv.tail(6).sum(); // Sum of x[3] to x[8]

    // Compute gradient if required
    if (!grad.empty()) {
        grad.assign(x.size(), 0.0);
        grad[0] = -optData->col(0);
        grad[1] = -optData->col(1);
        grad[2] = -optData->col(2);
        for (int i = 3; i < 9; ++i) {
            grad[i] = 1.0;
        }
    }

    return obj;
}

// Equality constraint function for the optimization problem
double equalityConstraint(const vector<double>& x, vector<double>& grad, void* data) {
    // Constraint: x[0] + x[1] + x[2] = 1
    if (!grad.empty()) {
        grad.assign(x.size(), 0.0);
        grad[0] = 1.0;
        grad[1] = 1.0;
        grad[2] = 1.0;
    }
    return x[0] + x[1] + x[2] - 1.0;
}
