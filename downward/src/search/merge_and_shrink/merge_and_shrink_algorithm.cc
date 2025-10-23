#include "merge_and_shrink_algorithm.h"

#include "distances.h"
#include "factored_transition_system.h"
#include "fts_factory.h"
#include "label_reduction.h"
#include "labels.h"
#include "merge_and_shrink_representation.h"
#include "merge_strategy.h"
#include "merge_strategy_factory.h"
#include "shrink_strategy.h"
#include "transition_system.h"
#include "types.h"
#include "utils.h"

#include "../plugins/plugin.h"
#include "../task_utils/task_properties.h"
#include "../utils/component_errors.h"
#include "../utils/countdown_timer.h"
#include "../utils/markup.h"
#include "../utils/math.h"
#include "../utils/system.h"
#include "../utils/timer.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <filesystem>

#include <nlohmann/json.hpp>
#include <fstream>

#include <cmath>     // for NaN/Inf handling if needed
#include <limits>    // for numeric_limits

using json = nlohmann::json;

using namespace std;
using plugins::Bounds;
using utils::ExitCode;

namespace merge_and_shrink {
static void log_progress(const utils::Timer &timer, const string &msg, utils::LogProxy &log) {
    log << "M&S algorithm timer: " << timer << " (" << msg << ")" << endl;
}
MergeAndShrinkAlgorithm::MergeAndShrinkAlgorithm(
    const shared_ptr<MergeStrategyFactory> &merge_strategy,
    const shared_ptr<ShrinkStrategy> &shrink_strategy,
    const shared_ptr<LabelReduction> &label_reduction,
    bool prune_unreachable_states, bool prune_irrelevant_states,
    int max_states, int max_states_before_merge,
    int threshold_before_merge, double main_loop_max_time,
    utils::Verbosity verbosity)
    : merge_strategy_factory(merge_strategy),
      shrink_strategy(shrink_strategy),
      label_reduction(label_reduction),
      max_states(max_states),
      max_states_before_merge(max_states_before_merge),
      shrink_threshold_before_merge(threshold_before_merge),
      prune_unreachable_states(prune_unreachable_states),
      prune_irrelevant_states(prune_irrelevant_states),
      log(utils::get_log_for_verbosity(verbosity)),
      main_loop_max_time(main_loop_max_time),
      starting_peak_memory(0) {
    handle_shrink_limit_defaults();
    // Asserting fields (not parameters).
    assert(this->max_states_before_merge >= 1);
    assert(this->max_states >= this->max_states_before_merge);
}

void MergeAndShrinkAlgorithm::handle_shrink_limit_defaults() {
    // If none of the two state limits has been set: set default limit.
    if (max_states == -1 && max_states_before_merge == -1) {
        max_states = 50000;
    }

    // If one of the max_states options has not been set, set the other
    // so that it imposes no further limits.
    if (max_states_before_merge == -1) {
        max_states_before_merge = max_states;
    } else if (max_states == -1) {
        if (utils::is_product_within_limit(
                max_states_before_merge, max_states_before_merge, INF)) {
            max_states = max_states_before_merge * max_states_before_merge;
        } else {
            max_states = INF;
        }
    }

    if (max_states_before_merge > max_states) {
        max_states_before_merge = max_states;
        if (log.is_warning()) {
            log << "WARNING: "
                << "max_states_before_merge exceeds max_states, "
                << "correcting max_states_before_merge." << endl;
        }
    }

    utils::verify_argument(max_states >= 1,
                           "Transition system size must be at least 1.");

    utils::verify_argument(max_states_before_merge >= 1,
                           "Transition system size before merge must be at least 1.");

    if (shrink_threshold_before_merge == -1) {
        shrink_threshold_before_merge = max_states;
    }

    utils::verify_argument(shrink_threshold_before_merge >= 1,
                           "Threshold must be at least 1.");

    if (shrink_threshold_before_merge > max_states) {
        shrink_threshold_before_merge = max_states;
        if (log.is_warning()) {
            log << "WARNING: "
                << "threshold exceeds max_states, "
                << "correcting threshold." << endl;
        }
    }
}

void MergeAndShrinkAlgorithm::report_peak_memory_delta(bool final) const {
    if (final)
        log << "Final";
    else
        log << "Current";
    log << " peak memory increase of merge-and-shrink algorithm: "
        << utils::get_peak_memory_in_kb() - starting_peak_memory << " KB"
        << endl;
}

void MergeAndShrinkAlgorithm::dump_options() const {
    if (log.is_at_least_normal()) {
        if (merge_strategy_factory) { // deleted after merge strategy extraction
            merge_strategy_factory->dump_options();
            log << endl;
        }

        log << "Options related to size limits and shrinking: " << endl;
        log << "Transition system size limit: " << max_states << endl
            << "Transition system size limit right before merge: "
            << max_states_before_merge << endl;
        log << "Threshold to trigger shrinking right before merge: "
            << shrink_threshold_before_merge << endl;
        log << endl;

        shrink_strategy->dump_options(log);
        log << endl;

        log << "Pruning unreachable states: "
            << (prune_unreachable_states ? "yes" : "no") << endl;
        log << "Pruning irrelevant states: "
            << (prune_irrelevant_states ? "yes" : "no") << endl;
        log << endl;

        if (label_reduction) {
            label_reduction->dump_options(log);
        } else {
            log << "Label reduction disabled" << endl;
        }
        log << endl;

        log << "Main loop max time in seconds: " << main_loop_max_time << endl;
        log << endl;
    }
}

void MergeAndShrinkAlgorithm::warn_on_unusual_options() const {
    string dashes(79, '=');
    if (!label_reduction) {
        if (log.is_warning()) {
            log << dashes << endl
                << "WARNING! You did not enable label reduction. " << endl
                << "This may drastically reduce the performance of merge-and-shrink!"
                << endl << dashes << endl;
        }
    } else if (label_reduction->reduce_before_merging() && label_reduction->reduce_before_shrinking()) {
        if (log.is_warning()) {
            log << dashes << endl
                << "WARNING! You set label reduction to be applied twice in each merge-and-shrink" << endl
                << "iteration, both before shrinking and merging. This double computation effort" << endl
                << "does not pay off for most configurations!"
                << endl << dashes << endl;
        }
    } else {
        if (label_reduction->reduce_before_shrinking() &&
            (shrink_strategy->get_name() == "f-preserving"
             || shrink_strategy->get_name() == "random")) {
            if (log.is_warning()) {
                log << dashes << endl
                    << "WARNING! Bucket-based shrink strategies such as f-preserving random perform" << endl
                    << "best if used with label reduction before merging, not before shrinking!"
                    << endl << dashes << endl;
            }
        }
        if (label_reduction->reduce_before_merging() &&
            shrink_strategy->get_name() == "bisimulation") {
            if (log.is_warning()) {
                log << dashes << endl
                    << "WARNING! Shrinking based on bisimulation performs best if used with label" << endl
                    << "reduction before shrinking, not before merging!"
                    << endl << dashes << endl;
            }
        }
    }

    if (!prune_unreachable_states || !prune_irrelevant_states) {
        if (log.is_warning()) {
            log << dashes << endl
                << "WARNING! Pruning is (partially) turned off!" << endl
                << "This may drastically reduce the performance of merge-and-shrink!"
                << endl << dashes << endl;
        }
    }
}

bool MergeAndShrinkAlgorithm::ran_out_of_time(
    const utils::CountdownTimer &timer) const {
    if (timer.is_expired()) {
        if (log.is_at_least_normal()) {
            log << "Ran out of time, stopping computation." << endl;
            log << endl;
        }
        return true;
    }
    return false;
}

// ============================================================================
// END: CORRECTED HELPER FUNCTION
// ============================================================================

void MergeAndShrinkAlgorithm::main_loop(
    FactoredTransitionSystem &fts,
    const TaskProxy &task_proxy) {

    utils::CountdownTimer timer(main_loop_max_time);
    if (log.is_at_least_normal()) {
        log << "Starting main loop ";
        if (main_loop_max_time == numeric_limits<double>::infinity()) {
            log << "without a time limit." << endl;
        } else {
            log << "with a time limit of " << main_loop_max_time << "s." << endl;
        }
    }

    int maximum_intermediate_size = 0;
    for (int i = 0; i < fts.get_size(); ++i) {
        int size = fts.get_transition_system(i).get_size();
        if (size > maximum_intermediate_size) {
            maximum_intermediate_size = size;
        }
    }

    if (label_reduction) {
        label_reduction->initialize(task_proxy);
    }

    unique_ptr<MergeStrategy> merge_strategy =
        merge_strategy_factory->compute_merge_strategy(task_proxy, fts);
    merge_strategy_factory = nullptr;

    auto log_main_loop_progress = [&timer, this](const string &msg) {
        log << "M&S algorithm main loop timer: "
            << timer.get_elapsed_time()
            << " (" << msg << ")" << endl;
    };

    // ✅ SETUP: Get output directories ONCE at the start
    std::string fd_output_dir = std::filesystem::absolute("fd_output").string();
    std::filesystem::create_directories(fd_output_dir);
    std::cout << "[M&S] Using fd_output: " << fd_output_dir << std::endl;

    int iteration = 0;

    while (fts.get_num_active_entries() > 1) {
        // ====================================================================
        // PHASE 1: GET NEXT MERGE PAIR FROM STRATEGY
        // ====================================================================

        pair<int, int> merge_indices = merge_strategy->get_next();
        if (ran_out_of_time(timer)) break;

        int merge_index1 = merge_indices.first;
        int merge_index2 = merge_indices.second;

        assert(merge_index1 != merge_index2);
        if (log.is_at_least_normal()) {
            log << "Next pair of indices: ("
                << merge_index1 << ", " << merge_index2 << ")" << endl;
            if (log.is_at_least_verbose()) {
                fts.statistics(merge_index1, log);
                fts.statistics(merge_index2, log);
            }
            log_main_loop_progress("after computation of next merge");
        }

        // ====================================================================
        // PHASE 2: LABEL REDUCTION (BEFORE SHRINKING)
        // ====================================================================

        bool reduced = false;
        if (label_reduction && label_reduction->reduce_before_shrinking()) {
            reduced = label_reduction->reduce(merge_indices, fts, log);
            if (log.is_at_least_normal() && reduced) {
                log_main_loop_progress("after label reduction");
            }
        }
        if (ran_out_of_time(timer)) break;

        // ====================================================================
        // PHASE 3: SHRINKING
        // ====================================================================

        bool shrunk = shrink_before_merge_step(
            fts,
            merge_index1,
            merge_index2,
            max_states,
            max_states_before_merge,
            shrink_threshold_before_merge,
            *shrink_strategy,
            log);
        if (log.is_at_least_normal() && shrunk) {
            log_main_loop_progress("after shrinking");
        }
        if (ran_out_of_time(timer)) break;

        // ====================================================================
        // PHASE 4: LABEL REDUCTION (BEFORE MERGING)
        // ====================================================================

        if (label_reduction && label_reduction->reduce_before_merging()) {
            reduced = label_reduction->reduce(merge_indices, fts, log);
            if (log.is_at_least_normal() && reduced) {
                log_main_loop_progress("after label reduction");
            }
        }
        if (ran_out_of_time(timer)) break;

        // ====================================================================
        // PHASE 5: EXPORT MERGE SIGNALS (BEFORE DATA) - SIMPLIFIED INLINE
        // ====================================================================

        {
            const auto& init1 = fts.get_distances(merge_index1).get_init_distances();
            const auto& goal1 = fts.get_distances(merge_index1).get_goal_distances();
            const auto& init2 = fts.get_distances(merge_index2).get_init_distances();
            const auto& goal2 = fts.get_distances(merge_index2).get_goal_distances();

            std::vector<int> f1(init1.size()), f2(init2.size());
            for (size_t i = 0; i < f1.size(); ++i) {
                f1[i] = (init1[i] == INF || goal1[i] == INF) ? INF : init1[i] + goal1[i];
            }
            for (size_t j = 0; j < f2.size(); ++j) {
                f2[j] = (init2[j] == INF || goal2[j] == INF) ? INF : init2[j] + goal2[j];
            }

            int ts1_transitions = 0;
            for (auto it = fts.get_transition_system(merge_index1).begin();
                 it != fts.get_transition_system(merge_index1).end(); ++it) {
                ts1_transitions += (*it).get_transitions().size();
            }

            int ts2_transitions = 0;
            for (auto it = fts.get_transition_system(merge_index2).begin();
                 it != fts.get_transition_system(merge_index2).end(); ++it) {
                ts2_transitions += (*it).get_transitions().size();
            }

            int ts1_size = (int)f1.size();
            int ts2_size = (int)f2.size();

            json product_mapping;
            for (int s = 0; s < ts1_size * ts2_size; ++s) {
                int s1 = s / ts2_size;
                int s2 = s % ts2_size;
                product_mapping[std::to_string(s)] = {{"s1", s1}, {"s2", s2}};
            }

            json before_data;
            before_data["ts1_id"] = merge_index1;
            before_data["ts2_id"] = merge_index2;
            before_data["iteration"] = iteration;
            before_data["ts1_f_values"] = f1;
            before_data["ts2_f_values"] = f2;
            before_data["ts1_size"] = ts1_size;
            before_data["ts2_size"] = ts2_size;
            before_data["ts1_num_transitions"] = ts1_transitions;
            before_data["ts2_num_transitions"] = ts2_transitions;
            before_data["product_mapping"] = product_mapping;

            // ✅ DIRECT WRITE: No helper function, just write it
            std::string before_path = fd_output_dir + "/merge_before_" + std::to_string(iteration) + ".json";
            std::ofstream before_file(before_path, std::ios::out | std::ios::trunc);
            if (!before_file.is_open()) {
                std::cerr << "[M&S] ERROR: Cannot create merge_before file: " << before_path << std::endl;
                throw std::runtime_error("Cannot create merge_before file");
            }
            before_file << before_data.dump(2);
            before_file.close();
            std::cout << "[M&S] ✅ Wrote merge_before_" << iteration << ".json" << std::endl;
        }

        // ====================================================================
        // PHASE 6: PERFORM ACTUAL MERGE
        // ====================================================================

        int merged_index = fts.merge(merge_index1, merge_index2, log);

        // ====================================================================
        // PHASE 7: EXPORT MERGE SIGNALS (AFTER DATA) - SIMPLIFIED INLINE
        // ====================================================================

        {
            const auto& init_dist = fts.get_distances(merged_index).get_init_distances();
            const auto& goal_dist = fts.get_distances(merged_index).get_goal_distances();

            std::vector<int> f_after(init_dist.size());
            for (size_t s = 0; s < f_after.size(); ++s) {
                f_after[s] = (init_dist[s] == INF || goal_dist[s] == INF) ? INF : init_dist[s] + goal_dist[s];
            }

            const TransitionSystem& merged_ts = fts.get_transition_system(merged_index);
            int num_goals = 0;
            for (size_t i = 0; i < f_after.size(); ++i) {
                if (merged_ts.is_goal_state(i)) num_goals++;
            }

            int merged_transitions = 0;
            for (auto it = merged_ts.begin(); it != merged_ts.end(); ++it) {
                merged_transitions += (*it).get_transitions().size();
            }

            // ✅ COMPUTE A* METRICS (inline, simple version)
            int nodes_expanded = 0;
            for (int i = 0; i < merged_ts.get_size(); ++i) {
                if (init_dist[i] != INF && goal_dist[i] != INF) {
                    nodes_expanded++;
                }
            }

            int reachable_states = nodes_expanded;
            double branching_factor = 1.0;
            if (reachable_states > 0 && merged_transitions > 0) {
                branching_factor = (double)merged_transitions / (double)reachable_states;
                if (std::isnan(branching_factor) || std::isinf(branching_factor) || branching_factor < 1.0) {
                    branching_factor = 1.0;
                }
            }

            int search_depth = 0;
            long long sum_goal_dist = 0;
            int reachable_goal_count = 0;
            for (int i = 0; i < merged_ts.get_size(); ++i) {
                if (init_dist[i] != INF && goal_dist[i] != INF) {
                    sum_goal_dist += goal_dist[i];
                    reachable_goal_count++;
                }
            }
            if (reachable_goal_count > 0) {
                search_depth = (int)std::round((double)sum_goal_dist / reachable_goal_count);
            }

            int best_goal_f = INF;
            for (int i = 0; i < merged_ts.get_size(); ++i) {
                if (merged_ts.is_goal_state(i) && init_dist[i] != INF && goal_dist[i] != INF) {
                    int f = init_dist[i] + goal_dist[i];
                    if (f < best_goal_f) {
                        best_goal_f = f;
                    }
                }
            }
            bool solution_found = (best_goal_f != INF);
            int solution_cost = solution_found ? best_goal_f : 0;

            json after_data;
            after_data["ts1_id"] = merge_index1;
            after_data["ts2_id"] = merge_index2;
            after_data["merged_id"] = merged_index;
            after_data["iteration"] = iteration;
            after_data["f_values"] = f_after;
            after_data["num_states"] = (int)f_after.size();
            after_data["num_goal_states"] = num_goals;
            after_data["num_transitions"] = merged_transitions;

            json search_signals;
            search_signals["nodes_expanded"] = nodes_expanded;
            search_signals["search_depth"] = search_depth;
            search_signals["solution_cost"] = solution_cost;
            search_signals["branching_factor"] = branching_factor;
            search_signals["solution_found"] = solution_found;
            after_data["search_signals"] = search_signals;

            // ✅ DIRECT WRITE: No helper function
            std::string after_path = fd_output_dir + "/merge_after_" + std::to_string(iteration) + ".json";
            std::ofstream after_file(after_path, std::ios::out | std::ios::trunc);
            if (!after_file.is_open()) {
                std::cerr << "[M&S] ERROR: Cannot create merge_after file: " << after_path << std::endl;
                throw std::runtime_error("Cannot create merge_after file");
            }
            after_file << after_data.dump(2);
            after_file.close();
            std::cout << "[M&S] ✅ Wrote merge_after_" << iteration << ".json" << std::endl;
        }

        // ====================================================================
        // PHASE 8: EXPORT MERGED TS JSON - SIMPLIFIED INLINE
        // ====================================================================

        {
            const TransitionSystem& ts = fts.get_transition_system(merged_index);

            json ts_json;
            ts_json["iteration"] = iteration;
            ts_json["num_states"] = ts.get_size();
            ts_json["init_state"] = ts.get_init_state();
            ts_json["transformed"] = (shrunk || reduced);

            std::vector<int> goal_states;
            for (int i = 0; i < ts.get_size(); ++i) {
                if (ts.is_goal_state(i)) {
                    goal_states.push_back(i);
                }
            }
            ts_json["goal_states"] = goal_states;
            ts_json["incorporated_variables"] = ts.get_incorporated_variables();

            std::vector<json> transitions;
            for (auto it = ts.begin(); it != ts.end(); ++it) {
                const auto& info = *it;
                const auto& label_group = info.get_label_group();
                const auto& trans_vec = info.get_transitions();

                for (int label : label_group) {
                    for (const auto& trans : trans_vec) {
                        transitions.push_back({
                            {"src", trans.src},
                            {"target", trans.target},
                            {"label", label}
                        });
                    }
                }
            }
            ts_json["transitions"] = transitions;

            std::string ts_path = fd_output_dir + "/ts_" + std::to_string(iteration) + ".json";

            // ✅ DIRECT WRITE: Simple, no helper function
            std::ofstream ts_file(ts_path, std::ios::out | std::ios::trunc);
            if (!ts_file.is_open()) {
                std::cerr << "[M&S] ERROR: Cannot create ts file: " << ts_path << std::endl;
                throw std::runtime_error("Cannot create ts file");
            }
            ts_file << ts_json.dump(2);
            ts_file.close();
            std::cout << "[M&S] ✅ Wrote ts_" << iteration << ".json with " << ts.get_size() << " states" << std::endl;
        }

        // ====================================================================
        // PHASE 9: INCREMENT ITERATION COUNTER
        // ====================================================================

        iteration++;  // ✅ MOVED HERE for clarity

        int abs_size = fts.get_transition_system(merged_index).get_size();
        if (abs_size > maximum_intermediate_size) {
            maximum_intermediate_size = abs_size;
        }

        if (log.is_at_least_normal()) {
            if (log.is_at_least_verbose()) {
                fts.statistics(merged_index, log);
            }
            log_main_loop_progress("after merging");
        }

        if (ran_out_of_time(timer)) {
            break;
        }

        // Pruning
        if (prune_unreachable_states || prune_irrelevant_states) {
            bool pruned = prune_step(
                fts,
                merged_index,
                prune_unreachable_states,
                prune_irrelevant_states,
                log);
            if (log.is_at_least_normal() && pruned) {
                if (log.is_at_least_verbose()) {
                    fts.statistics(merged_index, log);
                }
                log_main_loop_progress("after pruning");
            }
        }

        if (!fts.is_factor_solvable(merged_index)) {
            if (log.is_at_least_normal()) {
                log << "Abstract problem is unsolvable, stopping computation." << endl << endl;
            }
            break;
        }

        if (ran_out_of_time(timer)) {
            break;
        }

        if (log.is_at_least_verbose()) {
            report_peak_memory_delta();
        }
        if (log.is_at_least_normal()) {
            log << endl;
        }
    }

    log << "End of merge-and-shrink algorithm, statistics:" << endl;
    log << "Main loop runtime: " << timer.get_elapsed_time() << endl;
    log << "Maximum intermediate abstraction size: "
        << maximum_intermediate_size << endl;
    shrink_strategy = nullptr;
    label_reduction = nullptr;
}


FactoredTransitionSystem MergeAndShrinkAlgorithm::build_factored_transition_system(
    const TaskProxy &task_proxy) {
    if (starting_peak_memory) {
        cerr << "Calling build_factored_transition_system twice is not "
             << "supported!" << endl;
        utils::exit_with(utils::ExitCode::SEARCH_CRITICAL_ERROR);
    }
    starting_peak_memory = utils::get_peak_memory_in_kb();

    utils::Timer timer;
    log << "Running merge-and-shrink algorithm..." << endl;
    task_properties::verify_no_axioms(task_proxy);
    dump_options();
    warn_on_unusual_options();
    log << endl;

    const bool compute_init_distances =
        shrink_strategy->requires_init_distances() ||
        merge_strategy_factory->requires_init_distances() ||
        prune_unreachable_states;
    const bool compute_goal_distances =
        shrink_strategy->requires_goal_distances() ||
        merge_strategy_factory->requires_goal_distances() ||
        prune_irrelevant_states;
    FactoredTransitionSystem fts =
        create_factored_transition_system(
            task_proxy,
            compute_init_distances,
            compute_goal_distances,
            log);
    if (log.is_at_least_normal()) {
        log_progress(timer, "after computation of atomic factors", log);
    }

    /*
      Prune all atomic factors according to the chosen options. Stop early if
      one factor is unsolvable.

      TODO: think about if we can prune already while creating the atomic FTS.
    */
    bool pruned = false;
    bool unsolvable = false;
    for (int index = 0; index < fts.get_size(); ++index) {
        assert(fts.is_active(index));
        if (prune_unreachable_states || prune_irrelevant_states) {
            bool pruned_factor = prune_step(
                fts,
                index,
                prune_unreachable_states,
                prune_irrelevant_states,
                log);
            pruned = pruned || pruned_factor;
        }
        if (!fts.is_factor_solvable(index)) {
            log << "Atomic FTS is unsolvable, stopping computation." << endl;
            unsolvable = true;
            break;
        }
    }
    if (log.is_at_least_normal()) {
        if (pruned) {
            log_progress(timer, "after pruning atomic factors", log);
        }
        log << endl;
    }

    // ####################################################################################################
    // === Export Atomic Transition Systems ===
    {
        std::string filename = "merged_transition_systems.json";
        json all_ts;

        // Try to load existing merged systems if any
        std::ifstream infile(filename);
        if (infile) {
            infile >> all_ts;
            infile.close();
        }

        for (int i = 0; i < fts.get_size(); ++i) {
            if (!fts.is_active(i)) continue;

            const TransitionSystem& ts = fts.get_transition_system(i);

            json ts_json;
            ts_json["iteration"] = -1;  // Use -1 to mark atomic TS
            ts_json["num_states"] = ts.get_size();
            ts_json["init_state"] = ts.get_init_state();

            std::vector<int> goal_states;
            for (int j = 0; j < ts.get_size(); ++j)
                if (ts.is_goal_state(j))
                    goal_states.push_back(j);
            ts_json["goal_states"] = goal_states;

            ts_json["incorporated_variables"] = ts.get_incorporated_variables();

            std::vector<json> transitions;
            for (auto it = ts.begin(); it != ts.end(); ++it) {
                const auto& info = *it;
                const auto& label_group = info.get_label_group();
                const auto& trans_vec = info.get_transitions();

                for (int label : label_group) {
                    for (const auto& trans : trans_vec) {
                        transitions.push_back({
                            {"src", trans.src},
                            {"target", trans.target},
                            {"label", label}
                            });
                    }
                }
            }
            ts_json["transitions"] = transitions;

            all_ts.push_back(ts_json);
        }

        std::ofstream outfile(filename);
        outfile << all_ts.dump(4);  // Pretty print
        outfile.close();
    }
    // ####################################################################################################

    if (!unsolvable && main_loop_max_time > 0) {
        main_loop(fts, task_proxy);
    }
    const bool final = true;
    report_peak_memory_delta(final);
    log << "Merge-and-shrink algorithm runtime: " << timer << endl;
    log << endl;
    return fts;
}

void add_merge_and_shrink_algorithm_options_to_feature(plugins::Feature &feature) {
    // Merge strategy option.
    feature.add_option<shared_ptr<MergeStrategyFactory>>(
        "merge_strategy",
        "See detailed documentation for merge strategies. "
        "We currently recommend SCC-DFP, which can be achieved using "
        "{{{merge_strategy=merge_sccs(order_of_sccs=topological,merge_selector="
        "score_based_filtering(scoring_functions=[goal_relevance,dfp,total_order"
        "]))}}}");

    // Shrink strategy option.
    feature.add_option<shared_ptr<ShrinkStrategy>>(
        "shrink_strategy",
        "See detailed documentation for shrink strategies. "
        "We currently recommend non-greedy shrink_bisimulation, which can be "
        "achieved using {{{shrink_strategy=shrink_bisimulation(greedy=false)}}}");

    // Label reduction option.
    feature.add_option<shared_ptr<LabelReduction>>(
        "label_reduction",
        "See detailed documentation for labels. There is currently only "
        "one 'option' to use label_reduction, which is {{{label_reduction=exact}}} "
        "Also note the interaction with shrink strategies.",
        plugins::ArgumentInfo::NO_DEFAULT);

    // Pruning options.
    feature.add_option<bool>(
        "prune_unreachable_states",
        "If true, prune abstract states unreachable from the initial state.",
        "true");
    feature.add_option<bool>(
        "prune_irrelevant_states",
        "If true, prune abstract states from which no goal state can be "
        "reached.",
        "true");

    add_transition_system_size_limit_options_to_feature(feature);

    feature.add_option<double>(
        "main_loop_max_time",
        "A limit in seconds on the runtime of the main loop of the algorithm. "
        "If the limit is exceeded, the algorithm terminates, potentially "
        "returning a factored transition system with several factors. Also "
        "note that the time limit is only checked between transformations "
        "of the main loop, but not during, so it can be exceeded if a "
        "transformation is runtime-intense.",
        "infinity",
        Bounds("0.0", "infinity"));
}

tuple<shared_ptr<MergeStrategyFactory>, shared_ptr<ShrinkStrategy>,
      shared_ptr<LabelReduction>, bool, bool, int, int, int, double>
get_merge_and_shrink_algorithm_arguments_from_options(
    const plugins::Options &opts) {
    return tuple_cat(
        make_tuple(
            opts.get<shared_ptr<MergeStrategyFactory>>("merge_strategy"),
            opts.get<shared_ptr<ShrinkStrategy>>("shrink_strategy"),
            opts.get<shared_ptr<LabelReduction>>(
                "label_reduction", nullptr),
            opts.get<bool>("prune_unreachable_states"),
            opts.get<bool>("prune_irrelevant_states")),
        get_transition_system_size_limit_arguments_from_options(opts),
        make_tuple(opts.get<double>("main_loop_max_time"))
        );
}

void add_transition_system_size_limit_options_to_feature(plugins::Feature &feature) {
    feature.add_option<int>(
        "max_states",
        "maximum transition system size allowed at any time point.",
        "-1",
        Bounds("-1", "infinity"));
    feature.add_option<int>(
        "max_states_before_merge",
        "maximum transition system size allowed for two transition systems "
        "before being merged to form the synchronized product.",
        "-1",
        Bounds("-1", "infinity"));
    feature.add_option<int>(
        "threshold_before_merge",
        "If a transition system, before being merged, surpasses this soft "
        "transition system size limit, the shrink strategy is called to "
        "possibly shrink the transition system.",
        "-1",
        Bounds("-1", "infinity"));
}

tuple<int, int, int>
get_transition_system_size_limit_arguments_from_options(
    const plugins::Options &opts) {
    return make_tuple(
        opts.get<int>("max_states"),
        opts.get<int>("max_states_before_merge"),
        opts.get<int>("threshold_before_merge")
        );
}
}