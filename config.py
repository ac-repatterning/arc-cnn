"""
Module config.py
"""
import os


class Config:
    """
    Description
    -----------

    A class for configurations
    """

    def __init__(self) -> None:
        """
        <b>Notes</b><br>
        ------<br>

        Variables denoting a path - including or excluding a filename - have an underscore suffix; this suffix is
        excluded for names such as warehouse, storage, depository, *key, etc.<br><br>

        Over time, step-by-step, some of the items herein will be transferred to the overarching project's
        configurations zone/hub.

        """

        '''
        Keys
        '''
        self.architecture = 'arc-rnn-lstm'
        self.s3_parameters_key = 's3_parameters.yaml'
        self.arguments_key = f'architectures/{self.architecture}/arguments.json'
        self.metadata = f'architectures/{self.architecture}/metadata.json'


        '''
        Project Metadata
        '''
        self.project_tag = 'hydrography'
        self.project_key_name = 'HydrographyProject'


        '''
        Local Paths
        '''
        self.warehouse: str = os.path.join(os.getcwd(), 'warehouse')
        self.pathway_ = self.warehouse
