#pragma once

#include "merge_strategy.h"
#include "factored_transition_system.h"
#include "shrink_strategy.h"

#include <vector>
#include <memory>
#include <string>

namespace merge_and_shrink {

    class MergeStrategyGNN : public MergeStrategy {
    private:
        int iteration;
        std::string gnn_output_dir;
//        std::string fd_output_dir;

    public:
        explicit MergeStrategyGNN(const FactoredTransitionSystem& fts);

        std::pair<int, int> get_next() override;

        void set_next_merge_pair(
            const FactoredTransitionSystem&,
            const std::vector<std::shared_ptr<ShrinkStrategy>>&);
    };

}
