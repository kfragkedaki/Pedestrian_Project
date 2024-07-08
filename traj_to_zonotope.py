from src.reachability_analysis.simulation import get_test_config, get_test_label, get_cluster, get_initial_conditions, reachability_for_all_modes
from src.datasets.data import SVEAData
from src.transformer_model.model import create_model, evaluate
import json
import logging
import os
import rospy

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def __listener(self):
    """Subscribes to the topic containing only detected
    persons and applies the function __callback."""

    # TODO

    while not rospy.is_shutdown():
        rospy.spin()

def __callback(self, msg):
    """This method is a callback function that is triggered when a message is received.
    It interpolates person locations, stores the states of persons, and publishes
    the estimated person states to a 'person_state_estimation/person_states' topic.
    This implementation keeps sending the states of persons who have
    dropped out of frame, because the person might have dropped randomly.
    
    :param msg: message containing the detected persons
    :return: None"""

    # TODO


if __name__ == 'main':

    print('HELLO WORLD')

    config_path = '/home/sam/Desktop/Pedestrian_Project/experiments/SINDDataset_pretrained_2024-04-27_00-11-45_KIP/configuration.json'

    with open(config_path) as cnfg:
        config = json.load(cnfg)

    ROOT_RESOURCES = os.getcwd() + "/resources"


    config['original_data'] = False

    config_test = config.copy()
    config_test['online_data'] = True
    config_test['online_data'] = False
    config_test['pattern'] = None
    config_test['data_dir'] = '/home/sam/Desktop/Pedestrian_Project/bags'
    config_test['data_class'] = 'svea'

    data_oracle = SVEAData(config_test)

    #TODO trajectory = SVEAData.read_msg_and_update_df (include call to function that pads and chunks, simular to get_test_label inside of labeling_oracle.py)


    # Getting embeddings
    # TODO load trajectory into a DataLoader online_data
    # data_oracle = instance of SVEAData
    #  Create model, optimizer, and evaluators
    model, optimizer, trainer, val_evaluator, start_epoch = create_model(
        config, None, online_data, data_oracle, logger, device='cpu'
    )

    # Perform the evaluation
    aggr_metrics, embedding_data = evaluate(val_evaluator, config=config, save_embeddings=True)


    output = []
    # TODO iterate through trajectories
        # c = get_cluster(config, data_oracle) TODO can you batch with AnnoyModel?
        # test_cases = {f'c_{c}': f'Cluster: {c}'}
        # pos, v = get_initial_conditions(trajectory)
        # z, l, _b, _z = reachability_for_all_modes(pos=pos, vel=v, baseline=False, test_cases=test_cases, config=config, trajectory=trajectory, show_plot=True, save_plot=None)
        # output.append(z[0])

    # print(output)

    # TODO publish a list of zonotopes