#include "command_line.h"
#include "search_algorithm.h"

#include "tasks/root_task.h"
#include "task_utils/task_properties.h"
#include "utils/logging.h"
#include "utils/system.h"
#include "utils/timer.h"

#include <iostream>

//############################################################################################
#include "task_utils/causal_graph.h"
#include "tasks/root_task.h"
#include <fstream>
//############################################################################################


using namespace std;
using utils::ExitCode;

int main(int argc, const char **argv) {
    try {
        utils::register_event_handlers();

        if (argc < 2) {
            utils::g_log << usage(argv[0]) << endl;
            utils::exit_with(ExitCode::SEARCH_INPUT_ERROR);
        }

        bool unit_cost = false;
        if (static_cast<string>(argv[1]) != "--help") {
            utils::g_log << "reading input..." << endl;
            tasks::read_root_task(cin);
            utils::g_log << "done reading input!" << endl;
            TaskProxy task_proxy(*tasks::g_root_task);

            //############################################################################################
            causal_graph::CausalGraph cg(task_proxy);

            std::ofstream out("causal_graph.json");
            out << "{\n";

            // Write nodes
            out << "\"nodes\": [\n";
            const auto& variables = task_proxy.get_variables();
            for (size_t i = 0; i < variables.size(); ++i) {
                auto var = variables[i];
                out << "  { \"id\": " << var.get_id() << ", \"name\": \"" << var.get_name() << "\" }";
                if (i != variables.size() - 1) out << ",";
                out << "\n";
            }
            out << "],\n";

            // Write edges
            out << "\"edges\": [\n";
            bool first = true;
            for (auto src_var : variables) {
                int src_id = src_var.get_id();
                const std::vector<int>& dst_ids = cg.get_successors(src_id);
                for (int dst_id : dst_ids) {
                    if (!first) out << ",";
                    out << "\n  { \"from\": " << src_id << ", \"to\": " << dst_id << " }";
                    first = false;
                }
            }
            out << "\n]\n";
            out << "}\n";
            out.close();
            //############################################################################################

            unit_cost = task_properties::is_unit_cost(task_proxy);
        }

        shared_ptr<SearchAlgorithm> search_algorithm =
            parse_cmd_line(argc, argv, unit_cost);


        utils::Timer search_timer;
        search_algorithm->search();
        search_timer.stop();
        utils::g_timer.stop();

        search_algorithm->save_plan_if_necessary();
        search_algorithm->print_statistics();
        utils::g_log << "Search time: " << search_timer << endl;
        utils::g_log << "Total time: " << utils::g_timer << endl;

        ExitCode exitcode = search_algorithm->found_solution()
            ? ExitCode::SUCCESS
            : ExitCode::SEARCH_UNSOLVED_INCOMPLETE;
        exit_with(exitcode);
    } catch (const utils::ExitException &e) {
        /* To ensure that all destructors are called before the program exits,
           we raise an exception in utils::exit_with() and let main() return. */
        return static_cast<int>(e.get_exitcode());
    }
}
