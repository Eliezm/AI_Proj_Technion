fast_downward_plugin(
    NAME MERGE_AND_SHRINK
    HELP "Merge-and-shrink heuristic"
    SOURCES
        # Core algorithm
        distances.cc
        distances.h
        factored_transition_system.cc
        factored_transition_system.h
        fts_factory.cc
        fts_factory.h
        labels.cc
        labels.h
        label_reduction.cc
        label_reduction.h

        # Main algorithm (with integrated A* evaluator)
        merge_and_shrink_algorithm.cc
        merge_and_shrink_algorithm.h
        merge_and_shrink_heuristic.cc
        merge_and_shrink_heuristic.h
        merge_and_shrink_representation.cc
        merge_and_shrink_representation.h
        merge_quality_metrics.cc
        merge_quality_metrics.h
        merge_scoring_function.cc
        merge_scoring_function.h
        merge_scoring_function_dfp.cc
        merge_scoring_function_dfp.h
        merge_scoring_function_goal_relevance.cc
        merge_scoring_function_goal_relevance.h
        merge_scoring_function_miasm.cc
        merge_scoring_function_miasm.h
        merge_scoring_function_miasm_utils.cc
        merge_scoring_function_miasm_utils.h
        merge_scoring_function_single_random.cc
        merge_scoring_function_single_random.h
        merge_scoring_function_total_order.cc
        merge_scoring_function_total_order.h
        merge_selector.cc
        merge_selector.h
        merge_selector_score_based_filtering.cc
        merge_selector_score_based_filtering.h
        merge_strategy.cc
        merge_strategy.h
        merge_strategy_factory.cc
        merge_strategy_factory.h
        merge_strategy_precomputed.cc
        merge_strategy_precomputed.h
        merge_strategy_factory_precomputed.cc
        merge_strategy_factory_precomputed.h
        merge_strategy_factory_sccs.cc
        merge_strategy_factory_sccs.h
        merge_strategy_sccs.cc
        merge_strategy_sccs.h
        merge_strategy_factory_stateless.cc
        merge_strategy_factory_stateless.h
        merge_strategy_stateless.cc
        merge_strategy_stateless.h

        # GNN Strategy
        merge_strategy_factory_gnn.cc
        merge_strategy_factory_gnn.h
        merge_strategy_gnn.cc
        merge_strategy_gnn.h

        merge_tree.cc
        merge_tree.h
        merge_tree_factory.cc
        merge_tree_factory.h
        merge_tree_factory_linear.cc
        merge_tree_factory_linear.h
        shrink_bisimulation.cc
        shrink_bisimulation.h
        shrink_bucket_based.cc
        shrink_bucket_based.h
        shrink_fh.cc
        shrink_fh.h
        shrink_random.cc
        shrink_random.h
        shrink_strategy.cc
        shrink_strategy.h
        transition_system.cc
        transition_system.h
        types.cc
        types.h
        utils.cc
        utils.h
    DEPENDS PLUGINS
)