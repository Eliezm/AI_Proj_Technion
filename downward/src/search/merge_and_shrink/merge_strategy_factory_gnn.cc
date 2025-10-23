//#include "merge_strategy_factory_gnn.h"
//#include "merge_strategy_gnn.h"
//
//#include "../plugins/plugin.h"            // for TypedFeature
//#include "../utils/logging.h"             // brings in utils::Verbosity
//
//using namespace std;

#include "merge_strategy_factory_gnn.h"
#include "merge_strategy_gnn.h"

#include "../plugins/plugin.h"
#include "../utils/logging.h"

using std::string;
using std::unique_ptr;
using std::make_unique;
using std::shared_ptr;

namespace merge_and_shrink {

    // 1) Constructor: forward verbosity to base
    MergeStrategyFactoryGNN::MergeStrategyFactoryGNN(utils::Verbosity verbosity)
        : MergeStrategyFactory(verbosity) {
    }

    // 2) Instantiation
    unique_ptr<MergeStrategy> MergeStrategyFactoryGNN::compute_merge_strategy(
        const TaskProxy& /*task_proxy*/,
        const FactoredTransitionSystem& fts) {
        return make_unique<MergeStrategyGNN>(fts);
    }

    string MergeStrategyFactoryGNN::name() const {
        return "merge_gnn";
    }

    void MergeStrategyFactoryGNN::dump_strategy_specific_options() const {
        // No extra options here
    }

    // 3) Plugin registration, exactly like the others
    class MergeStrategyFactoryGNNFeature
        : public plugins::TypedFeature<MergeStrategyFactory, MergeStrategyFactoryGNN> {
    public:
        MergeStrategyFactoryGNNFeature()
            : TypedFeature("merge_gnn") {
            document_title("GNN-based merge strategy");
            document_synopsis(
                "Reads the next merge pair from an external JSON file "
                "produced by a GNN.");
            add_merge_strategy_options_to_feature(*this);
        }

        // This unpacks (verbosity) from opts into the ctor
        shared_ptr<MergeStrategyFactoryGNN> create_component(
            const plugins::Options& opts) const override {
            return plugins::make_shared_from_arg_tuples<MergeStrategyFactoryGNN>(
                get_merge_strategy_arguments_from_options(opts)
            );
        }
    };

    // static registration
    static plugins::FeaturePlugin<MergeStrategyFactoryGNNFeature> _plugin;

}  // namespace merge_and_shrink
