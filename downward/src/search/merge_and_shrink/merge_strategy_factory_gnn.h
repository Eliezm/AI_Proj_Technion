#ifndef MERGE_AND_SHRINK_MERGE_STRATEGY_FACTORY_GNN_H
#define MERGE_AND_SHRINK_MERGE_STRATEGY_FACTORY_GNN_H

#include "merge_strategy_factory.h"

namespace merge_and_shrink {

    class MergeStrategyFactoryGNN : public MergeStrategyFactory {
    public:
        // Match the other factories: take verbosity
        explicit MergeStrategyFactoryGNN(utils::Verbosity verbosity);

        // Create the strategy object
        std::unique_ptr<MergeStrategy> compute_merge_strategy(
            const TaskProxy& task_proxy,
            const FactoredTransitionSystem& fts) override;

        std::string name() const override;
        void dump_strategy_specific_options() const override;
        bool requires_init_distances()  const override { return false; }
        bool requires_goal_distances()  const override { return false; }
    };

}  // namespace merge_and_shrink

#endif
