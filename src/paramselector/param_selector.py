# Example parameter selector. Just outputs given commands in order
class ParamSelector:
    def __init__(this, parsel_config_filename=None, model_config_filename=None, error_config_filename=None):
        this.commands_list = []
        this.commands_ind = 0
        if parsel_config_filename is not None:
            f = open(parsel_config_filename)
            while True:
                run_model_command = f.readline().rstrip('\n')
                run_analyze_command = f.readline().rstrip('\n')
                if not run_analyze_command:
                    break
                this.commands_list.append([run_model_command, run_analyze_command])

    def should_continue(this):
        return this.commands_ind < len(this.commands_list)

    def next_commands(this):
        i = this.commands_ind
        this.commands_ind += 1
        return this.commands_list[i][0], this.commands_list[i][1]

    def read_analysis(this, analysis):
        pass
