#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <mutex>
#include <cmath>
#include <numeric>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

struct TradeData {
    double tsla_price;
    double nio_price;
    double spread;
    double zscore;
    int signal;  // 1 = Long, -1 = Short, 0 = No Position
    double factor;
};

vector<TradeData> load_data(const string &filename) {
    vector<TradeData> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    getline(file, line); // Skip header
    while (getline(file, line)) {
        stringstream ss(line);
        TradeData trade;
        string value;
        vector<string> row;
        
        while (getline(ss, value, ',')) {
            row.push_back(value);
        }

        if (row.size() < 6) continue; // Ensure valid row

        trade.tsla_price = stod(row[0]);
        trade.nio_price = stod(row[1]);
        trade.spread = stod(row[2]);
        trade.zscore = stod(row[3]);
        trade.signal = stoi(row[4]);
        trade.factor = stod(row[5]);

        data.push_back(trade);
    }
    
    file.close();
    return data;
}

struct BacktestResult {
    double total_return = 0.0;
    double max_drawdown = 0.0;
    double sharpe_ratio = 0.0;
    int total_trades = 0;
};

BacktestResult run_backtest(const vector<TradeData> &data, double capital, double risk_per_trade) {
    double balance = capital;
    double max_balance = capital;
    double min_balance = capital;
    double returns = 0.0;
    int trades = 0;

    double position_size = capital * risk_per_trade;
    vector<double> pnl_list;
    
    for (size_t i = 1; i < data.size(); i++) {
        if (data[i].signal != 0) {
            trades++;
            double entry_price = (data[i].tsla_price + data[i].nio_price) / 2;
            double exit_price = (data[i-1].tsla_price + data[i-1].nio_price) / 2;

            double pnl = (exit_price - entry_price) * data[i].signal * position_size;
            balance += pnl;
            pnl_list.push_back(pnl);

            max_balance = max(max_balance, balance);
            min_balance = min(min_balance, balance);
        }
    }

    // Sharpe Ratio
    double mean_pnl = accumulate(pnl_list.begin(), pnl_list.end(), 0.0) / pnl_list.size();
    double variance = 0.0;
    for (double pnl : pnl_list) {
        variance += pow(pnl - mean_pnl, 2);
    }
    variance /= pnl_list.size();
    double stddev = sqrt(variance);
    double sharpe_ratio = (stddev > 0) ? (mean_pnl / stddev) * sqrt(252) : 0.0;

    BacktestResult result;
    result.total_return = (balance - capital) / capital * 100;
    result.max_drawdown = (max_balance - min_balance) / max_balance * 100;
    result.sharpe_ratio = sharpe_ratio;
    result.total_trades = trades;

    return result;
}

int main() {
    string filename = "..data/signals.csv";
    vector<TradeData> data = load_data(filename);

    if (data.empty()) {
        cerr << "No data - xiting..." << endl;
        return 1;
    }

    double initial_capital = 100000;
    double risk_per_trade = 0.02;

    // Multithreaded Backtesting
    mutex result_mutex;
    vector<thread> threads;
    vector<BacktestResult> results(4);

    for (int i = 0; i < 4; i++) {
        threads.emplace_back([&, i]() {
            auto result = run_backtest(data, initial_capital, risk_per_trade);
            lock_guard<mutex> lock(result_mutex);
            results[i] = result;
        });
    }

    for (auto &t : threads) t.join();

    // Aggregate results
    double total_return = 0, max_drawdown = 0, avg_sharpe = 0;
    int total_trades = 0;

    for (const auto &r : results) {
        total_return += r.total_return;
        max_drawdown = max(max_drawdown, r.max_drawdown);
        avg_sharpe += r.sharpe_ratio;
        total_trades += r.total_trades;
    }
    avg_sharpe /= results.size();

    cout << "Return: " << total_return << endl;
    cout << "Max Drawdown: " << max_drawdown << endl;
    cout << "Sharpe Ratio: " << avg_sharpe << endl;
    cout << "Total Trades: " << total_trades << endl;

    return 0;
}
