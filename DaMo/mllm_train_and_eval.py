class Trainer:
    def train(self, model_path, training_dataset, data_mixture_proportion):
        """
        The MLLM training framework can use open-source frameworks like llama-factory, etc., which are not repeated here.
        During training, different training datasets need to be mixed according to data_mixture_proportion.
        """
        pass

class Evaluator:
    def eval(self, ckpt_path, downstream_tasks):
        """
        Evaluate the downstream tasks performance of the ckpt saved during training. For phoneAgentBench, see the relevant instructions in the readme. For other open-source benches, please refer to the relevant open-source projects, which are not detailed here.
        We provide some experimental data, see processed_data_random_50.xlsx for reference.
        """
        pass