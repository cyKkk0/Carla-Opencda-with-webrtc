# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import carla
import asyncio
import threading
import opencda.scenario_testing.utils.sim_api as sim_api
from opencda.core.common.cav_world import CavWorld
from opencda.scenario_testing.evaluations.evaluate_manager import \
    EvaluationManager
from opencda.scenario_testing.utils.yaml_utils import add_current_time


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


async def add_control_channel(single_cav, webrtc_server):
    
    pass


def run_scenario(opt, scenario_params, webrtc_server=None, webrtc_client=None, server_loop=None, client_loop=None):
    try:
        scenario_params = add_current_time(scenario_params)

        # create CAV world
        cav_world = CavWorld(opt.apply_ml)

        # create scenario manager
        scenario_manager = sim_api.ScenarioManager(scenario_params,
                                                   opt.apply_ml,
                                                   opt.version,
                                                   town='Town06',
                                                   cav_world=cav_world,
                                                   webrtc_server=webrtc_server,
                                                   webrtc_client=webrtc_client,
                                                   server_loop=server_loop,
                                                   client_loop=client_loop)

        if opt.record:
            scenario_manager.client. \
                start_recorder("single_town06_carla.log", True)

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'])

        # create background traffic in carla
        traffic_manager, bg_veh_list = \
            scenario_manager.create_traffic_carla()

        # create evaluation manager
        eval_manager = \
            EvaluationManager(scenario_manager.cav_world,
                              script_name='single_2lanefree_carla',
                              current_time=scenario_params['current_time'])

        spectator = scenario_manager.world.get_spectator()
        # for i, single_cav in enumerate(single_cav_list):
        #     future = asyncio.run_coroutine_threadsafe(add_control_channel(single_cav, webrtc_server), server_loop)
        #     future.result()
        # run steps
        while True:
            scenario_manager.tick()
            transform = single_cav_list[0].vehicle.get_transform()
            spectator.set_transform(carla.Transform(
                transform.location +
                carla.Location(
                    z=50),
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

