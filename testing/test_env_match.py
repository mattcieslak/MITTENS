import sys
sys.path.append("..")
from mittens.env_match import EnvironmentMatch
FIB_FILE="DSI-Studio/sub-ABCD_acq-noisefree_space-T1w_desc-preproc_space-T1w_MAPMRI.fib"

def test_env_match():
    matcher = EnvironmentMatch(reconstruction=FIB_FILE,
                               fixel_threshold=0,
                               cutoff_value=0.3
                               )
    similarity_graph = matcher.environment_similarity_graph()

test_env_match()
