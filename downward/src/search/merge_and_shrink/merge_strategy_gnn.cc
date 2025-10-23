// FILE: downward/src/search/merge_and_shrink/merge_strategy_gnn.cc

#include "merge_strategy_gnn.h"
#include "../utils/system.h"

#include <nlohmann/json.hpp>
#include <fstream>
#include <thread>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cstdlib>

using namespace std;
using json = nlohmann::json;
using namespace merge_and_shrink;

MergeStrategyGNN::MergeStrategyGNN(const FactoredTransitionSystem& fts)
    : MergeStrategy(fts),
      iteration(0) {
    cout << "\n[GNN::INIT] MergeStrategyGNN initialized\n" << endl;
}

// ============================================================================
// ✅ COMPLETELY REFACTORED: Robust cross-platform path resolution
// ============================================================================
std::filesystem::path get_project_root() {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "[GNN::PATH] Current working directory: " << cwd.string() << std::endl;

    // If we're already in downward/, go up one level
    if (cwd.filename() == "downward") {
        cwd = cwd.parent_path();
        std::cout << "[GNN::PATH] Detected cwd is downward/, adjusting to parent" << std::endl;
    }

    // Search upward, starting from cwd
    std::filesystem::path search_path = cwd;

    for (int i = 0; i < 15; ++i) {
        std::filesystem::path downward_subdir = search_path / "downward";

        if (std::filesystem::is_directory(downward_subdir)) {
            std::filesystem::path gnn_output_dir = downward_subdir / "gnn_output";
            std::filesystem::path fd_output_dir = downward_subdir / "fd_output";

            if (std::filesystem::is_directory(gnn_output_dir) &&
                std::filesystem::is_directory(fd_output_dir)) {
                std::cout << "[GNN::PATH] ✅ Found project root: " << search_path.string() << std::endl;
                return search_path.string();
            }
        }

        std::filesystem::path parent = search_path.parent_path();
        if (parent == search_path) break;
        search_path = parent;
    }

    std::cout << "[GNN::PATH] ⚠️ Could not find project root, defaulting to cwd" << std::endl;
    return cwd.string();
}

std::pair<int, int> MergeStrategyGNN::get_next() {
    int idx = iteration;

    // ✅ FIX: Use std::filesystem for ALL path operations (cross-platform!)
    std::filesystem::path project_root_str = get_project_root();
    std::filesystem::path project_root = std::filesystem::absolute(project_root_str);

    std::filesystem::path gnn_input_dir = project_root / "downward" / "gnn_output";
    std::filesystem::path fd_output_dir = project_root / "downward" / "fd_output";

    std::filesystem::path in_path = gnn_input_dir / ("merge_" + std::to_string(idx) + ".json");
    std::filesystem::path ack_path = fd_output_dir / ("gnn_ack_" + std::to_string(idx) + ".json");

    std::cout << "\n[GNN::ITERATION " << idx << "] ========================================" << std::endl;
    std::cout << "[GNN::ITERATION " << idx << "] Waiting for GNN merge decision..." << std::endl;
    std::cout << "[GNN::ITERATION " << idx << "] Input file:  " << in_path.string() << std::endl;
    std::cout << "[GNN::ITERATION " << idx << "] ACK file:    " << ack_path.string() << std::endl;

    // ✅ WAIT FOR INPUT FILE with timeout
    const int sleep_ms = 50;  // 50ms polling (fast feedback)
    const int max_wait_ms = 600000;  // 10 minutes timeout
    int elapsed_ms = 0;

    while (!std::filesystem::exists(in_path)) {
        // Log progress every 10 seconds
        if (elapsed_ms > 0 && elapsed_ms % 10000 == 0) {
            std::cout << "[GNN::ITERATION " << idx << "] ... waiting (" << elapsed_ms / 1000 << "s) ..."
                      << std::endl;
        }

        if (elapsed_ms > max_wait_ms) {
            std::cerr << "\n[GNN::ERROR] ❌ TIMEOUT: No merge decision received!" << std::endl;
            std::cerr << "[GNN::ERROR] Expected file: " << in_path.string() << std::endl;
            std::cerr << "[GNN::ERROR] Timeout: " << max_wait_ms / 1000 << " seconds" << std::endl;

            // Diagnostic logging
            std::cout << "[GNN::DIAG] Files in " << gnn_input_dir.string() << ":" << std::endl;
            try {
                if (std::filesystem::exists(gnn_input_dir)) {
                    for (const auto& entry : std::filesystem::directory_iterator(gnn_input_dir)) {
                        std::cout << "[GNN::DIAG]   - " << entry.path().filename().string() << std::endl;
                    }
                } else {
                    std::cout << "[GNN::DIAG]   (directory doesn't exist)" << std::endl;
                }
            } catch (const std::exception& e) {
                std::cout << "[GNN::DIAG] Error listing directory: " << e.what() << std::endl;
            }

            throw std::runtime_error("GNN merge decision timeout: " + in_path.string());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        elapsed_ms += sleep_ms;
    }

    std::cout << "[GNN::ITERATION " << idx << "] ✅ Merge decision file found!" << std::endl;

    // ✅ READ MERGE DECISION WITH VALIDATION
    json merge_decision;
    try {
        std::ifstream fin(in_path);
        if (!fin.is_open()) {
            throw std::runtime_error("Cannot open merge decision file: " + in_path.string());
        }

        fin >> merge_decision;
        fin.close();

        // Validate JSON structure
        if (!merge_decision.contains("merge_pair")) {
            throw std::runtime_error("Missing 'merge_pair' key in merge decision");
        }
        if (!merge_decision["merge_pair"].is_array() ||
            merge_decision["merge_pair"].size() != 2) {
            throw std::runtime_error("'merge_pair' must be an array of exactly 2 integers");
        }

        std::cout << "[GNN::ITERATION " << idx << "] ✅ Successfully parsed merge decision JSON"
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[GNN::ERROR] Failed to read/parse merge decision: " << e.what() << std::endl;
        throw;
    }

    int u = merge_decision["merge_pair"][0].get<int>();
    int v = merge_decision["merge_pair"][1].get<int>();

    std::cout << "[GNN::ITERATION " << idx << "] Merge decision: (" << u << ", " << v << ")"
              << std::endl;

    // ✅ WRITE ACKNOWLEDGMENT ATOMICALLY
    try {
        // Ensure output directory exists
        std::filesystem::create_directories(fd_output_dir);

        json ack_data;
        ack_data["iteration"] = idx;
        ack_data["merge_pair"] = {u, v};
        ack_data["received"] = true;
        ack_data["timestamp"] = std::time(nullptr);

        // Write to temporary file first
        std::filesystem::path temp_path = ack_path.string() + ".tmp";
        {
            std::ofstream fout(temp_path);
            if (!fout.is_open()) {
                throw std::runtime_error("Cannot open temp ACK file: " + temp_path.string());
            }

            fout << ack_data.dump(2);  // Pretty print
            fout.flush();

            // Explicit close to ensure all data is written
            fout.close();
            if (fout.fail()) {
                throw std::runtime_error("Failed to write or close temp ACK file");
            }
        }

        // Atomic rename: temp → final
        try {
            std::filesystem::rename(temp_path, ack_path);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "[GNN::WARNING] Failed to rename ACK file: " << e.what() << std::endl;
            // Try direct write as fallback
            std::ofstream fout(ack_path);
            if (fout.is_open()) {
                fout << ack_data.dump(2);
                fout.close();
            }
        }

        std::cout << "[GNN::ITERATION " << idx << "] ✅ Wrote ACK to: " << ack_path.string()
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[GNN::WARNING] Could not write ACK file: " << e.what() << std::endl;
        // Don't throw—we still got the merge decision
    }

    iteration++;

    std::cout << "[GNN::ITERATION " << idx << "] ✅ Returning merge pair (" << u << ", " << v
              << ")" << std::endl;
    std::cout << "[GNN::ITERATION " << idx << "] ========================================\n"
              << std::endl;

    return {u, v};
}

void MergeStrategyGNN::set_next_merge_pair(
    const FactoredTransitionSystem&,
    const std::vector<std::shared_ptr<ShrinkStrategy>>&) {
    // Not used
}