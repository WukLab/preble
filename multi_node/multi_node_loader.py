from gpu_stats_profiling import get_gpu_profile
from model_runtime_manager import ModelDetails
from typing import DefaultDict, List
from collections import defaultdict
from model_runtime_manager import ExtendedSGLangRuntime, EndpointRuntimeInterface
import signal
import sys

class GPUConfig:
    def __init__(self, gpu_id, url=None, use_ssh=False, ssh_config={}) -> None:
        self.gpu_id = gpu_id
        self.url = url
        self.use_ssh = use_ssh
        self.ssh_config = ssh_config

class MultiNodeLoader:
    def __init__(self) -> None:
        self.models_allocated = []
        self.gpus_to_model_allocated: DefaultDict[int, List[ModelDetails]] = (
            defaultdict(list)
        )
        self.current_gpu_memory_usage = get_gpu_profile()
        signal.signal(signal.SIGINT, self.runtime_cleanup_handler)

    def runtime_cleanup_handler(self, sig, frame):
        print('You pressed Ctrl+C! Shutting down all remote servers...')
        for instance in self.models_allocated:
            for runtime_instance in instance.runtimes:
                runtime_instance.shutdown()
        sys.exit(0)

    def load_model(self, model_path, gpu_configs=[], **kwargs) -> ModelDetails:
        """
        Load a model onto the specified gpus

        Note: Could manage this directly in python but SGLang uses global variables
        There's also a question on how to unload memory
        """
        model_details = ModelDetails(model_path, gpu_configs)
        model_details.load_runtimes(model_path=model_path, gpu_configs=gpu_configs, **kwargs)
        # TODO verify if the memory is available
        self.models_allocated.append(model_details)
        # for gpu in , urls=url:
        #     self.gpus_to_model_allocated[gpu].append(model_details)
        return model_details

    def unload_model(self, model_details: ModelDetails):
        """
        Unload a model from the gpus
        """
        for runtime in model_details.runtimes:
            runtime.shutdown()
        if model_details in self.models_allocated:
            self.models_allocated.remove(model_details)

        # for gpu in model_details.gpus:
        #     self.gpus_to_model_allocated[gpu].remove(model_details)
        #     self.update_gpu_memory_usage(gpu)
        model_details.runtimes = []
        # model_details.gpus = []
        return model_details

    def update_gpu_memory_usage(self, gpu):
        self.current_gpu_memory_usage[gpu] = get_gpu_profile()[gpu]