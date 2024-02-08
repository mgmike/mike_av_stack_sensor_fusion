from setuptools import setup
import os
from glob import glob

package_name = 'mike_av_stack_sensor_fusion'
tracking = 'mike_av_stack_sensor_fusion/tracking'
detection = 'mike_av_stack_sensor_fusion/detection'
tools = 'mike_av_stack_sensor_fusion/tools/ros_conversions'
ros2_numpy = 'mike_av_stack_sensor_fusion/ros2_numpy/ros2_numpy'

setup(
    name=package_name,
    version='0.0.2',
    packages=[package_name,
              tracking, 
              detection, 
              tools,
              ros2_numpy],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'configs'), glob('mike_av_stack_sensor_fusion/configs/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.*')),
        (os.path.join('share', package_name, 'weights'), glob('mike_av_stack_sensor_fusion/weights/*')),
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
