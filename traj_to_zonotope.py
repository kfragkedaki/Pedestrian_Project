
from src.reachability_analysis.simulation import get_test_config, get_test_label, get_cluster, get_initial_conditions, reachability_for_all_modes
from src.reachability_analysis.labeling_oracle import LabelingOracleSVEAData
from src.datasets.data import SVEAData, ROSData, SINDData
from src.transformer_model.model import create_model, evaluate
import json
import logging
import os
import rospy
from torch.utils.data import DataLoader
from src.datasets.data import data_factory, Normalizer
from src.datasets.masked_datasets import collate_unsuperv
from src.utils.load_data import load_task_datasets
from src.clustering.NearestNeighbor import AnnoyModel
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class TrajToZonotope:

    def __init__(self):
        self.config_path = '/home/sam/Desktop/Pedestrian_Project/config-ros.json'

        with open(self.config_path) as cnfg:
            self.config = json.load(cnfg)

        ROOT_RESOURCES = os.getcwd() + "/resources"

        self.config['original_data'] = False
        self.config['online_data'] = True
        self.config['pattern'] = None
        self.config['data_dir'] = '/home/sam/Desktop/Pedestrian_Project/bags'
        self.config['data_class'] = 'svea'
        self.config['eval_only'] = True
        self.config['val_ratio'] = 1
        self.config['output_dir'] = 'experiments/ROS_experiment_2024-07-08_16-51-06_FOz/eval'

        self.data_oracle = ROSData(self.config)
        self.nn_model = AnnoyModel(config=self.config)


    @staticmethod
    def zonotope_to_rectangle(z):
        """
        Convert a zonotope to its bounding rectangle.
        
        Parameters:
        z (zonotope): The zonotope to convert.

        Returns:
        (tuple): Top left and bottom right coordinates of the rectangle.
        """
        x = z.x
        G = z.G
        
        # Sum of the absolute values of the generators
        extent = np.sum(np.abs(G), axis=1)
        
        # Top left and bottom right coordinates
        top_left = x - extent
        bottom_right = x + extent
        
        return (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1])


    def listener(self):
        """Subscribes to the topic containing only detected
        persons and applies the function __callback."""

        while not rospy.is_shutdown():
            rospy.spin()

    def callback(self, msg):
        """This method is a callback function that is triggered when a message is received.
        It interpolates person locations, stores the states of persons, and publishes
        the estimated person states to a 'person_state_estimation/person_states' topic.
        This implementation keeps sending the states of persons who have
        dropped out of frame, because the person might have dropped randomly.
        
        :param msg: message containing the detected persons
        :return: None"""

        # self.data_oracle.process_message(msg)
        val_data = self.data_oracle.feature_df

        # Pre-process features
        if self.config["data_normalization"] != "none":
            logger.info("Normalizing data ...")
            normalizer = Normalizer(self.config["data_normalization"])
            val_data = normalizer.normalize(val_data)

        task_dataset, collate_fn = load_task_datasets(self.config)
        val_dataset = task_dataset(self.data_oracle.feature_df, [1])

        # Dataloaders
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
            collate_fn=lambda x: collate_fn(x, max_len=self.data_oracle.max_seq_len),
        )

        model, optimizer, trainer, val_evaluator, start_epoch = create_model(
            self.config, None, val_loader, self.data_oracle, logger, device='cpu'
        )

        aggr_metrics, embedding_data = evaluate(val_evaluator, config=self.config, save_embeddings=True)


        output = []
        labeling_oracle = LabelingOracleSVEAData(self.config)
        for embedding, target in zip(embedding_data['embeddings'], embedding_data['targets']):
            c = self.nn_model.get(embedding)
            test_cases = {f'c_{c}': f'Cluster: {c}'}
            pos, v = get_initial_conditions(target)
            z, l, _b, _z = reachability_for_all_modes(pos=pos, vel=v, baseline=False, test_cases=test_cases, config=self.config, trajectory=target, show_plot=False, save_plot=None, _sind = labeling_oracle)
            output.append(TrajToZonotope.zonotope_to_rectangle(z[0]))

        print(output)


if __name__ == "__main__":
    
    traj_to_zonotope = TrajToZonotope()
    traj_to_zonotope.callback('msg')

