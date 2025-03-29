# -*- coding: utf-8 -*-
"""
Scenario testing: Single vehicle dring in the customized 2 lane highway map.
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import os

import carla

import opencda.scenario_testing.utils.sim_api as sim_api
import opencda.scenario_testing.utils.customized_map_api as map_api

from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import \
    add_current_time


def control_to_dict(control):
    """
    将 carla.VehicleControl 对象转换为字典
    """
    return {
        'throttle': control.throttle,
        'steer': control.steer,
        'brake': control.brake,
        'hand_brake': control.hand_brake,
        'reverse': control.reverse,
        'manual_gear_shift': control.manual_gear_shift,
        'gear': control.gear
    }

def dict_to_control(control_dict):
    """
    将字典转换为 carla.VehicleControl 对象
    """
    return carla.VehicleControl(
        throttle=control_dict.get('throttle', 0.0),
        steer=control_dict.get('steer', 0.0),
        brake=control_dict.get('brake', 0.0),
        hand_brake=control_dict.get('hand_brake', False),
        reverse=control_dict.get('reverse', False),
        manual_gear_shift=control_dict.get('manual_gear_shift', False),
        gear=control_dict.get('gear', 0)
    )

def run_scenario(opt, scenario_params):
    try:
        scenario_params = add_current_time(scenario_params)
        current_path = os.path.dirname(os.path.realpath(__file__))
        xodr_path = os.path.join(
            current_path,
            '../assets/2lane_freeway_simplified/2lane_freeway_simplified.xodr')

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)
        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   xodr_path=xodr_path,
                                                   cav_world=cav_world)

        if opt.record:
            scenario_manager.client. \
                start_recorder("single_2lanefree_carla.log", True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'],
                                                    map_helper=map_api.
                                                    spawn_helper_2lanefree)

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='single_2lanefree_carla',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()
        # run steps
        while True:
            scenario_manager.tick()
            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location +
                carla.Location(
                    z=70),
                carla.Rotation(
                    pitch=-
                    90)))

            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                control = single_cav.run_step()
                data_dict = control_to_dict(control)
                data_ctrl = dict_to_control(data_dict)
                single_cav.vehicle.apply_control(data_ctrl)

    finally:
        eval_manager.evaluate()

        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.close()

        for v in single_cav_list:
            v.destroy()
        for v in bg_veh_list:
            v.destroy()
