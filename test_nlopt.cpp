#include <iostream>
#include <nlopt.hpp>

double myfunc(const std::vector<double> &x, std::vector<double> &grad, void *my_func_data) {
    double a = x[0];
    double b = x[1];
    return a * a + b * b;  // 優化目標：簡單的二次函數
}

int main() {
    // 定義問題的維度
    nlopt::opt opt(nlopt::LD_LBFGS, 2);

    // 設置目標函數
    opt.set_min_objective(myfunc, NULL);

    // 設置變量的範圍
    std::vector<double> lb(2, -1.0);
    opt.set_lower_bounds(lb);

    // 設置停止條件
    opt.set_xtol_rel(1e-4);

    // 初始化參數
    std::vector<double> x(2, 0.5);  // 初始猜測值
    double minf;  // 儲存優化的最小值

    try {
        // 優化
        nlopt::result result = opt.optimize(x, minf);
        std::cout << "找到最小值" << minf << "，參數為 [" << x[0] << ", " << x[1] << "]\n";
    }
    catch (std::exception &e) {
        std::cerr << "優化失敗：" << e.what() << std::endl;
    }

    return 0;
}

