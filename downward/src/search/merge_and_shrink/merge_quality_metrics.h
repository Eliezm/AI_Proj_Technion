#pragma once

#include <vector>
#include <map>
#include <string>
#include <nlohmann/json.hpp>

namespace merge_and_shrink {

using json = nlohmann::json;

/**
 * MergeQualityMetrics: Computes detailed quality metrics for each merge operation.
 * This data is exported to JSON and read by Python for reward computation.
 */
class MergeQualityMetrics {
public:
    // Information preservation metrics
    float f_value_stability_score;  // How much F-values changed [0,1]
    int num_states_with_significant_f_change;  // Count of states where F changed >threshold
    float avg_f_value_change;  // Average absolute F-value change
    float max_f_value_change;  // Max absolute F-value change

    // State space metrics
    int states_before;
    int states_after;
    int delta_states;  // Increase in state space

    // Reachability metrics
    int unreachable_states_created;  // New PRUNED_STATE entries
    int reachable_states_reduced;

    // Goal-reachability metrics
    int goal_unreachable_states_created;
    float goal_reachability_ratio;  // Fraction of states that can reach goal

    // Path quality metrics (detecting shortcuts)
    int new_shortest_paths_created;  // Paths that became shorter
    float avg_path_length_change;  // Average path length delta
    int states_with_shorter_path_from_init;

    // Transition system structure metrics
    int num_transitions_before;
    int num_transitions_after;
    float transition_density_change;  // Edges per state

    // Shrink quality metrics
    int shrink_equivalence_classes;  // Number of classes created by bisimulation
    float shrink_quality_score;  // How good was the shrink [0,1]

    // Label reduction effect
    int labels_before;
    int labels_after;
    float label_reduction_ratio;

    // Composite score for quick assessment
    float overall_merge_quality;  // Weighted combination [0,1]

    json to_json() const;
    static MergeQualityMetrics from_json(const json& j);
};

}