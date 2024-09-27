#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <nlopt.hpp>
#include <algorithm>

using namespace std;
using namespace Eigen;

// 定義全局變量以便在優化函數中使用
VectorXd col(3);
MatrixXd SM;
MatrixXd DM;
VectorXd W(9);

// 函數聲明
MatrixXd YuScore(const MatrixXd& DM);
MatrixXd YuScore2(const MatrixXd& DM);
double Hellin(double a, double b, double c, double d, double e, double f, double g, double h);
vector<int> Rank(const VectorXd& input);
double NLP(const vector<double>& x, vector<double>& grad, void* my_func_data);

// 新增約束條件函數
double constraint_eq(const vector<double>& x, vector<double>& grad, void* data);

int main() {
    // 初始化 DM
    DM = MatrixXd(3,12);
    DM << 0.95, 1.0, 0.0, 0.0,   0.0, 0.0, 0.95, 1.0,    0.0, 0.0, 0.95, 1.0,
          0.90, 1.0, 0.0, 0.0,   0.0, 0.0, 0.90, 1.0,    0.0, 0.0, 0.90, 1.0,
          0.85, 1.0, 0.0, 0.0,   0.0, 0.0, 0.85, 1.0,    0.0, 0.0, 0.85, 1.0;

    // 初始化 IVIFW
    MatrixXd IVIFW(3,4);
    IVIFW << 0.5, 0.5, 0.5, 0.5,
             0.5, 0.5, 0.5, 0.5,
             0.5, 0.5, 0.5, 0.5;

    // 步驟 1：計算得分矩陣 SM 和猶豫度 H
    SM = YuScore(DM);
    MatrixXd H = YuScore2(DM);

    // 初始化 x0 向量，長度為 9，初始值為 0.5
    std::vector<double> x0(9, 0.5);


    // 步驟 2：設置優化問題的參數
    VectorXd lb = VectorXd::Zero(9);
    VectorXd ub = VectorXd::Ones(9);

    // 提取 ivifv
    VectorXd ivifv11 = DM.row(0).segment(0,4);
    VectorXd ivifv21 = DM.row(1).segment(0,4);
    VectorXd ivifv31 = DM.row(2).segment(0,4);

    VectorXd ivifv12 = DM.row(0).segment(4,4);
    VectorXd ivifv22 = DM.row(1).segment(4,4);
    VectorXd ivifv32 = DM.row(2).segment(4,4);

    VectorXd ivifv13 = DM.row(0).segment(8,4);
    VectorXd ivifv23 = DM.row(1).segment(8,4);
    VectorXd ivifv33 = DM.row(2).segment(8,4);

    // 計算 Hellinger 距離
    double dis11_21 = Hellin(ivifv11(0), ivifv11(1), ivifv11(2), ivifv11(3),
                             ivifv21(0), ivifv21(1), ivifv21(2), ivifv21(3));
    double dis11_31 = Hellin(ivifv11(0), ivifv11(1), ivifv11(2), ivifv11(3),
                             ivifv31(0), ivifv31(1), ivifv31(2), ivifv31(3));
    double dis21_31 = Hellin(ivifv21(0), ivifv21(1), ivifv21(2), ivifv21(3),
                             ivifv31(0), ivifv31(1), ivifv31(2), ivifv31(3));
    double col1 = round((dis11_21 + dis11_31 + dis21_31) / 3.0 * 1000) / 1000.0;

    double dis12_22 = Hellin(ivifv12(0), ivifv12(1), ivifv12(2), ivifv12(3),
                             ivifv22(0), ivifv22(1), ivifv22(2), ivifv22(3));
    double dis12_32 = Hellin(ivifv12(0), ivifv12(1), ivifv12(2), ivifv12(3),
                             ivifv32(0), ivifv32(1), ivifv32(2), ivifv32(3));
    double dis22_32 = Hellin(ivifv22(0), ivifv22(1), ivifv22(2), ivifv22(3),
                             ivifv32(0), ivifv32(1), ivifv32(2), ivifv32(3));
    double col2 = round((dis12_22 + dis12_32 + dis22_32) / 3.0 * 1000) / 1000.0;

    double dis13_23 = Hellin(ivifv13(0), ivifv13(1), ivifv13(2), ivifv13(3),
                             ivifv23(0), ivifv23(1), ivifv23(2), ivifv23(3));
    double dis13_33 = Hellin(ivifv13(0), ivifv13(1), ivifv13(2), ivifv13(3),
                             ivifv33(0), ivifv33(1), ivifv33(2), ivifv33(3));
    double dis23_33 = Hellin(ivifv23(0), ivifv23(1), ivifv23(2), ivifv23(3),
                             ivifv33(0), ivifv33(1), ivifv33(2), ivifv33(3));
    double col3 = round((dis13_23 + dis13_33 + dis23_33) / 3.0 * 1000) / 1000.0;

    col << col1, col2, col3;

    cout << "col1 = " << col1 << endl;
    cout << "col2 = " << col2 << endl;
    cout << "col3 = " << col3 << endl;

   // 步驟 3：優化求解
    nlopt::opt local_opt(nlopt::LD_MMA, 9); // 局部優化算法 LD_MMA
    nlopt::opt opt(nlopt::AUGLAG, 9);       // 主算法 AUGLAG 用於等式約束

    opt.set_local_optimizer(local_opt); // 設置局部優化器

    vector<double> lb_vec(lb.data(), lb.data() + lb.size());
    vector<double> ub_vec(ub.data(), ub.data() + ub.size());
    opt.set_lower_bounds(lb_vec);
    opt.set_upper_bounds(ub_vec);

    opt.set_min_objective(NLP, NULL);

    // 添加等式約束
    opt.add_equality_constraint(constraint_eq, NULL, 1e-8);

    // 設置容忍度
    opt.set_xtol_rel(1e-4);  // 調整容忍度
    opt.set_maxeval(500);    // 限制最大迭代次數

    // 初始值
    vector<double> x(x0.data(), x0.data() + x0.size());

    double minf;
    try {
        nlopt::result result = opt.optimize(x, minf);
    } catch (std::exception& e) {
        cout << "優化失敗：" << e.what() << endl;
        return -1;
    }

    // 輸出結果
    cout << "W = [";
    for(int i = 0; i < x.size(); ++i) {
        W(i) = round(x[i]*1000)/1000.0;
        cout << W(i);
        if(i != x.size()-1) cout << ", ";
    }
    cout << "]" << endl;

    // 步驟 4：計算加權得分
    int m = SM.rows();
    int n = SM.cols();
    VectorXd WS(m);

    for(int i = 0; i < m; ++i) {
        double score = 0;
        for(int j = 0; j < n; ++j) {
            score += W(j) * SM(i,j);
        }
        cout << "==============================================" << endl;
        cout << "WS(" << i+1 << ")" << endl;
        cout << "= ";
        for(int j = 0; j < n-1; ++j) {
            cout << W(j) << " * " << SM(i,j) << " + ";
        }
        cout << W(n-1) << " * " << SM(i,n-1) << endl;

        WS(i) = round(score * 1000) / 1000.0;
        cout << "= 未四捨五入 " << score << " = 四捨五入後 " << WS(i) << endl;
    }

    // 排序
    vector<int> PO = Rank(WS);

    // 輸出排名結果
    cout << "PO = [";
    for(size_t i = 0; i < PO.size(); ++i) {
        cout << PO[i];
        if(i != PO.size()-1) cout << ", ";
    }
    cout << "]" << endl;

    return 0;
}

// YuScore 函數實現
MatrixXd YuScore(const MatrixXd& DM) {
    int m = DM.rows();
    int n = DM.cols() / 4;
    MatrixXd SM(m, n);

    auto s = [](double a, double b, double c, double d) {
        double value = 0.5*(a - c + b - d)
            + (sin(0.5*M_PI*a) + sin(0.5*M_PI*b)) / (1 + sin(0.5*M_PI*c) + sin(0.5*M_PI*d))
            + (cos(0.5*M_PI*c) + cos(0.5*M_PI*d)) / (1 + cos(0.5*M_PI*a) + cos(0.5*M_PI*b))
            + 1;
        return round(value * 1000) / 1000.0;
    };

    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            double a = DM(i, j*4 + 0);
            double b = DM(i, j*4 + 1);
            double c = DM(i, j*4 + 2);
            double d = DM(i, j*4 + 3);
            SM(i, j) = s(a, b, c, d);

            cout << "SM(" << i+1 << ", " << j+1 << ") = ("
                 << a << " - " << c << " + " << b << " - " << d << ")/2 + (sin(0.5*pi*"
                 << a << ") + sin(0.5*pi*" << b << ")) / (1 + sin(0.5*pi*"
                 << c << ") + sin(0.5*pi*" << d << ")) + (cos(0.5*pi*"
                 << c << ") + cos(0.5*pi*" << d << ")) / (1 + cos(0.5*pi*"
                 << a << ") + cos(0.5*pi*" << b << ")) + 1 = " << SM(i, j) << endl;
        }
    }

    return SM;
}

// YuScore2 函數實現
MatrixXd YuScore2(const MatrixXd& DM) {
    int m = DM.rows();
    int n = DM.cols() / 4;
    MatrixXd H(m, n*2);

    int count = 1;
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            double a = DM(i, j*4 + 0);
            double b = DM(i, j*4 + 1);
            double c = DM(i, j*4 + 2);
            double d = DM(i, j*4 + 3);

            double s1_value = 1 - b - d;
            double s2_value = 1 - a - c;

            cout << "h(" << count << ")= [1 - " << b << " - " << d << ", 1 - " << a << " - " << c << "]"
                 << " = [" << s1_value << ", " << s2_value << "]" << endl;

            H(i, j*2) = s1_value;
            H(i, j*2 + 1) = s2_value;
            count++;
        }
    }

    return H;
}

// Hellinger 距離函數實現
double Hellin(double a, double b, double c, double d, double e, double f, double g, double h) {
    double s1 = sqrt(a) - sqrt(e);
    double s2 = sqrt(b) - sqrt(f);
    double s3 = sqrt(c) - sqrt(g);
    double s4 = sqrt(d) - sqrt(h);
    double s5 = sqrt(1 - b - d) - sqrt(1 - f - h);
    double s6 = sqrt(1 - a - c) - sqrt(1 - e - g);

    double dis = 0.5 * sqrt(s1*s1 + s2*s2 + s3*s3 + s4*s4 + s5*s5 + s6*s6);
    return dis;
}

// 排序函數實現
vector<int> Rank(const VectorXd& input) {
    vector<pair<double, int>> vec;
    for(int i = 0; i < input.size(); ++i) {
        vec.push_back(make_pair(input(i), i+1));
    }

    sort(vec.begin(), vec.end(), greater<pair<double, int>>());

    vector<int> ranks(input.size());
    int rank = 1;
    for(size_t i = 0; i < vec.size(); ++i) {
        if(i > 0 && vec[i].first < vec[i-1].first) {
            rank = i + 1;
        }
        ranks[vec[i].second - 1] = rank;
    }

    return ranks;
}

// 優化目標函數
double NLP(const vector<double>& x, vector<double>& grad, void* my_func_data) {
    // 將 x 轉換為 Eigen 的 VectorXd
    VectorXd xv = VectorXd::Map(x.data(), x.size());

    double obj = -(col(0)*xv(0) + col(1)*xv(1) + col(2)*xv(2))
                 + xv(3) + xv(4) + xv(5) + xv(6) + xv(7) + xv(8);

    // 如果需要計算梯度（NLopt 需要）
    if (!grad.empty()) {
        grad.assign(x.size(), 0.0);
        grad[0] = -col(0);
        grad[1] = -col(1);
        grad[2] = -col(2);
        for (int i = 3; i < 9; ++i) {
            grad[i] = 1.0;
        }
    }

    return obj;
}

// 等式約束函數
double constraint_eq(const vector<double>& x, vector<double>& grad, void* data) {
    // x[0] + x[1] + x[2] = 1
    if (!grad.empty()) {
        grad.assign(x.size(), 0.0);
        grad[0] = 1.0;
        grad[1] = 1.0;
        grad[2] = 1.0;
    }
    return x[0] + x[1] + x[2] - 1.0;
}
