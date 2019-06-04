# Example parameter selector. Does ternary search for best parameters
class ParamSelector:
    def __init__(self, parsel_config_filename=None, model_config_filename=None, error_config_filename=None):
        self.model_config_filename = model_config_filename
        self.commands = None
        if parsel_config_filename is not None:
            f = open(parsel_config_filename)
            run_model_command = f.readline().rstrip('\n')
            run_analyze_command = f.readline().rstrip('\n')
            if run_analyze_command:
                self.commands = (run_model_command, run_analyze_command)
        self.low = 0.001
        self.high = 1
        self.eps = 0.0001

    def should_continue(self):
        return self.high - self.low > self.eps

    def next_commands(self):
        left_mid = (2*self.low + self.high) / 3
        right_mid = (self.low + 2*self.high) / 3
        if self.model_config_filename is not None:
            params = "{\"multinomial_nb__alpha\": [" + left_mid + ", " + right_mid + "]}"
            f = open(self.model_config_filename, 'w')
            f.write(params)
        return self.commands[0], self.commands[1]

    def read_analysis(self, analysis):
        # TODO: check from analysis whether left_mid or right_mid was better
        pass
