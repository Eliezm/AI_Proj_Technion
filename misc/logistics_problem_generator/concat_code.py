#!/usr/bin/env python3
import os
import glob

OUTPUT_FILE = "finishing_reward_function.txt"

def get_py_files():
    """
    Returns a list of all .py files in cwd, excluding this script itself.
    """
    all_py = glob.glob("*.py")
    this_script = os.path.basename(__file__)
    return [f for f in all_py if f != this_script]

def concatenate(files, out_path):
    with open(out_path, "w", encoding="utf-8") as out:
        for fname in files:
            out.write(f"The file {fname} code is this:\n")
            # out.write(f"Please understand how it is compatible with the rest of my codebase for the mission i asked you.\n")

            try:
                with open(fname, "r", encoding="utf-8") as inp:
                    out.write(inp.read())
            except Exception as e:
                out.write(f"[Error reading {fname}: {e}]\n")
            out.write(f"\n")
            # out.write(f"End: The file {fname} code is this:\n")
            out.write("\n" + "-" * 80 + "\n\n")

def main():
    # py_files = get_py_files()
    # py_files =[
    #     "run_all_baselines.py",
    #     "run_intra_domain_generalization.py",
    #     "run_scale_generalization.py",
    #     "train_final_model.py",
    #     "run_cross_domain_generalization.py",
    #     "analyze_results.py",
    # ]
    # py_files = [
    #     "gnn_model.py",
    #     "gnn_policy.py",
    #     "graph_tracker.py",
    #     "merge_env.py",
    #     "run_all_baselines.py",
    #     "run_intra_domain_generalization.py",
    #     "run_scale_generalization.py",
    #     "train_final_model.py",
    #     "run_cross_domain_generalization.py",
    #     "analyze_results.py",
    #     "validate_baselines.py",
    #     "validate_cross.py",
    #     "validate_intra.py",
    #     "validation_sweep.py",
    # ]
    # py_files = [
    #     "evaluation.py",
    #     "random_policy.py",
    #     "default_policy.py",
    #     "validation_sweep.py",
    # ]

    # py_files = [
    #     "gnn_model.py",
    #     "gnn_policy.py",
    #     "graph_tracker.py",
    #     "merge_env.py",
    #     "run_all_baselines.py",
    #     "run_intra_domain_generalization.py",
    #     "run_scale_generalization.py",
    #     "train_final_model.py",
    #     "run_cross_domain_generalization.py",
    #     "analyze_results.py",
    #     "validate_baselines.py",
    #     "validate_cross.py",
    #     "validate_intra.py",
    #     "evaluation.py",
    #     "validation_sweep.py",
    #     "common_utils.py",
    #     "default_policy.py",
    #     "hyperparameter_sweep.py",
    #     "random_policy.py",
    #     "merge_env_helper.py",
    #
    #     "reward_function_variants.py",
    #     "reward_info_extractor.py",
    #     "test_integrated_reward.py",
    #     "test_reward_extraction.py",
    #     "test_reward_functions.py",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.cc",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.h",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.h",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy_gnn.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy_gnn.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy.cc",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_representation.h",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_representation.cc",
    # ]

    # py_files = [ # for completing the reward function
    #     "reward_function_variants.py",
    #     "reward_info_extractor.py",
    #     "test_integrated_reward.py",
    #     "test_reward_extraction.py",
    #     "test_reward_functions.py",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.cc",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.h",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.h",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy_gnn.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy_gnn.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy.cc",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_representation.h",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_representation.cc",
    #     "merge_env.py",
    #     "merge_env_helper.py",
    # ]

    # py_files = [
    # # Core GNN and Environment Files
    # "gnn_model.py",
    # "gnn_policy.py",
    # "graph_tracker.py",
    # "merge_env.py",
    # "merge_env_helper.py",
    # "common_utils.py",
    #
    # # Reward Function Implementation
    # "reward_function_variants.py",
    # "reward_info_extractor.py",
    #
    # # Experiment and Training Scripts
    # "train_final_model.py",
    # "run_all_baselines.py",
    # "run_cross_domain_generalization.py",
    # "run_intra_domain_generalization.py",
    # "run_scale_generalization.py",
    # "hyperparameter_sweep.py",
    #
    # # Evaluation and Analysis
    # "analyze_results.py",
    # "evaluation.py",
    # "default_policy.py",
    # "random_policy.py",
    #
    # # Validation Scripts
    # "validation_sweep.py",
    # "validate_baselines.py",
    # "validate_cross.py",
    # "validate_intra.py",
    #
    # # Test Files
    # "test_integrated_reward.py",
    # "test_reward_extraction.py",
    # "test_reward_functions.py",
    #
    # # C++ Fast Downward Source Files
    # "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.cc",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.h",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.cc",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.h",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_representation.cc",
    # "downward/src/search/merge_and_shrink/merge_and_shrink_representation.h",
    # "downward/src/search/merge_and_shrink/merge_strategy.cc",
    # "downward/src/search/merge_and_shrink/merge_strategy.h",
    # "downward/src/search/merge_and_shrink/merge_strategy_factory.cc",
    # "downward/src/search/merge_and_shrink/merge_strategy_factory.h",
    # "downward/src/search/merge_and_shrink/merge_strategy_gnn.cc",
    # "downward/src/search/merge_and_shrink/merge_strategy_gnn.h",
    # "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.cc",
    # "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.h",
    # ]

    # py_files = [
    #     # Core GNN and Environment Files
    #     "gnn_model.py",
    #     "gnn_policy.py",
    #     "graph_tracker.py",
    #     "merge_env.py",
    #     "common_utils.py",
    #
    #     # Reward Function Implementation
    #     "reward_function_variants.py",
    #     "reward_info_extractor.py",
    #
    #     # Experiment and Training Scripts
    #     "train_final_model.py",
    #     "run_all_baselines.py",
    #     "run_cross_domain_generalization.py",
    #     "run_intra_domain_generalization.py",
    #     "run_scale_generalization.py",
    #     "hyperparameter_sweep.py",
    #     "run_generalization.py",
    #
    #     # Evaluation and Analysis
    #     "analyze_results.py",
    #     "evaluation.py",
    #     "default_policy.py",
    #     "random_policy.py",
    #
    #     # Validation Scripts
    #     "validation_sweep.py",
    #     "validate_baselines.py",
    #     "validate_cross.py",
    #     "validate_intra.py",
    #
    #     # Test Files
    #     "test_integrated_reward.py",
    #     "test_reward_extraction.py",
    #     "test_reward_functions.py",
    #
    #     # C++ Fast Downward Source Files
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.cc",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.h",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.cc",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.h",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_representation.cc",
    #     "downward/src/search/merge_and_shrink/merge_and_shrink_representation.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy_gnn.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy_gnn.h",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.cc",
    #     "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.h",
    # ]

    # infra_files = [
    #     ### experiments
    #     # "shared_experiment_utils.py",
    #     # "overfit_experiment_final.py",
    #     # "experiment_2_problem_generalization.py",
    #     # "experiment_3_scale_generalization.py",
    #     # "curriculum_learning_final.py",
    #     #
    #     # # Core GNN and Environment Files
    #     # "gnn_model.py",
    #     # "gnn_policy.py",
    #     # "graph_tracker.py",
    #     # "merge_env.py",
    #     # "common_utils.py",
    #     #
    #     # # Reward Function Implementation
    #     # "reward_function_variants.py",
    #     # "reward_info_extractor.py",
    #     #
    #     # # evaluation
    #     # "analysis_and_visualization.py",
    #     # "evaluation_comprehensive.py",
    #     # "run_full_evaluation.py",
    #     #
    #     #
    #     # # "evaluation.py",
    #     # "default_policy.py",
    #     # "random_policy.py",
    #     #
    #     # "train_real_working.py",
    #
    #     # problem generation
    #     "problem_generator.py",
    #     "domain_generators.py",
    #     "baseline_validator.py",
    #     "generator_utils.py",
    #
    #     # C++ Fast Downward Source Files
    #     # "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.cc",
    #     # "downward/src/search/merge_and_shrink/merge_and_shrink_algorithm.h",
    #     # "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.cc",
    #     # "downward/src/search/merge_and_shrink/merge_and_shrink_heuristic.h",
    #     # "downward/src/search/merge_and_shrink/merge_and_shrink_representation.cc",
    #     # "downward/src/search/merge_and_shrink/merge_and_shrink_representation.h",
    #     # "downward/src/search/merge_and_shrink/merge_strategy.cc",
    #     # "downward/src/search/merge_and_shrink/merge_strategy.h",
    #     # "downward/src/search/merge_and_shrink/merge_strategy_factory.cc",
    #     # "downward/src/search/merge_and_shrink/merge_strategy_factory.h",
    #     # "downward/src/search/merge_and_shrink/merge_strategy_gnn.cc",
    #     # "downward/src/search/merge_and_shrink/merge_strategy_gnn.h",
    #     # "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.cc",
    #     # "downward/src/search/merge_and_shrink/merge_strategy_factory_gnn.h",
    #
    #     # "experiment_config.py",
    #
    #
    #
    #     # "prepare_dataset_debug.py",
    #     # "train_mvp_debug.py",
    #     # "test_environment_debug.py",
    #     # "toy_data_setup_debug.py",
    #     # "debug_training.py",
    #     # "train_mvp_real.py",
    #     # "complete_train_validation_small.py",
    #     # "baseline_benchmarking.py",
    #
    #
    #
    #
    #     # experiments
    #     # "experiment_1_problem_overfit.py",
    #     # "problem_generalization_experiment.py",
    #     # "scale_generalization_experiment.py",
    #     # "experiment_4_curriculum_learning.py",
    #
    #
    #
    #
    #
    # ]

    infra_files = [
        "config.py",
        "state.py",
        "actions.py",
        "goal_archetypes.py",
        "goal_validators.py",
        "backward_generator.py",
        "pddl_writer.py",
        "baseline_planner.py",
        "metadata_store.py",
        "validator.py",
        "main.py",
        "__init__.py",
        "logistics_problem_builder.py",
        "example_usage.py",
        "problem_validator.py",
    ]

    if not infra_files:
        print("No Python files found to concatenate.")
        return
    concatenate(infra_files, OUTPUT_FILE)
    print(f"✅ Concatenated {len(infra_files)} file(s) into '{OUTPUT_FILE}'")

    # py_files = get_py_files()
    # for p in py_files:
    #     print(f"{p}")

if __name__ == "__main__":
    main()



