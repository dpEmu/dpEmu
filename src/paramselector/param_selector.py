# Example parameter selector. Just outputs given commands in order
class ParamSelector:
    def __init__(self, parsel_config_filename=None, model_config_filename=None, error_config_filename=None):
        self.commands_list = []
        self.commands_ind = 0
        if parsel_config_filename is not None:
            f = open(parsel_config_filename)
            while True:
                run_model_command = f.readline().rstrip('\n')
                run_analyze_command = f.readline().rstrip('\n')
                if not run_analyze_command:
                    break
                self.commands_list.append([run_model_command, run_analyze_command])

    def should_continue(self):
        return self.commands_ind < len(self.commands_list)

    def next_commands(self):
        i = self.commands_ind
        self.commands_ind += 1
        return self.commands_list[i][0], self.commands_list[i][1]

    def read_analysis(self, analysis):
        pass
