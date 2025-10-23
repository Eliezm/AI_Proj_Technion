#include "merge_quality_metrics.h"

namespace merge_and_shrink {

json MergeQualityMetrics::to_json() const {
    json j;
    j["f_value_stability_score"] = f_value_stability_score;
    j["num_significant_f_changes"] = num_states_with_significant_f_change;
    j["avg_f_value_change"] = avg_f_value_change;
    j["max_f_value_change"] = max_f_value_change;
    j["states_before"] = states_before;
    j["states_after"] = states_after;
    j["delta_states"] = delta_states;
    j["unreachable_states_created"] = unreachable_states_created;
    j["reachable_states_reduced"] = reachable_states_reduced;
    j["goal_unreachable_states_created"] = goal_unreachable_states_created;
    j["goal_reachability_ratio"] = goal_reachability_ratio;
    j["new_shortest_paths_created"] = new_shortest_paths_created;
    j["avg_path_length_change"] = avg_path_length_change;
    j["states_with_shorter_path_from_init"] = states_with_shorter_path_from_init;
    j["num_transitions_before"] = num_transitions_before;
    j["num_transitions_after"] = num_transitions_after;
    j["transition_density_change"] = transition_density_change;
    j["shrink_equivalence_classes"] = shrink_equivalence_classes;
    j["shrink_quality_score"] = shrink_quality_score;
    j["labels_before"] = labels_before;
    j["labels_after"] = labels_after;
    j["label_reduction_ratio"] = label_reduction_ratio;
    j["overall_merge_quality"] = overall_merge_quality;
    return j;
}

}