from transformers import AutoTokenizer
import random
import sys, os


# Add the parent directory of the 'src' directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from e2e_loogle_config import exp_args as loogle_exp_args
from multi_experiment_benchmark_utils import AllExperiments, ExperimentType, DefaultWorkload, ConfigurableMajorExperimentArgs

exp_args = AllExperiments(
    loogle_exp_args.experiments
)