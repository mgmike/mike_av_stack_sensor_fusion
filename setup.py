from setuptools import setup
import os
from glob import glob

package_name = 'mike_av_stack_sensor_fusion'
tracking = 'mike_av_stack_sensor_fusion/tracking'
detection = 'mike_av_stack_sensor_fusion/detection'
fpn_resnet_models = 'mike_av_stack_sensor_fusion/detection/objdet_models/fpn_resnet/models'
fpn_resnet_utils = 'mike_av_stack_sensor_fusion/detection/objdet_models/fpn_resnet/utils'
fpn_resnet_pretrained = 'mike_av_stack_sensor_fusion/detection/objdet_models/fpn_resnet/pretrained'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name, 
              tracking, 
              detection, 
              fpn_resnet_models, 
              fpn_resnet_utils, 
              fpn_resnet_pretrained],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'configs'), glob('mike_av_stack_sensor_fusion/configs/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mike',
    maintainer_email='mgmike1023@aol.com',
    description='Michael\'s autonomous vehicle stack built on top of carla ros bridge',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_fusion = mike_av_stack_sensor_fusion.sensor_fusion:main',

        ],
    },
)
