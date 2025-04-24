# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import carla
import asyncio
import pickle
import time
import threading
import multiprocessing
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


async def add_control_channel(single_cav):
    
    pass


def recv_control(recv_pipe, single_cav):
    control_dict = pickle.loads(recv_pipe.recv())
    control = dict_to_control(control_dict)
    # print(control)
    single_cav.vehicle.apply_control(control)


def run_scenario(opt, scenario_params, Webrtc_server=None, Webrtc_client=None):
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
                                                   Webrtc_server=Webrtc_server,
                                                   Webrtc_client=Webrtc_client)

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
        # be careful, avoid using the same port twice when simulating several vehicles
        port = 8099
        webrtc_server = Webrtc_server('127.0.0.1', port)
        webrtc_client = Webrtc_client('127.0.0.1', port)
        parent_conn = {}
        recv_ch_conn = {}
        for i in range(len(single_cav_list)):
            parent_conn[f'vehicle{i}'], child_conn = multiprocessing.Pipe()
            recv_fa_conn, recv_ch_conn[f'vehicle{i}'] = multiprocessing.Pipe()
            _ = webrtc_server.run_server_in_new_process(add_data=True,recv_pipe=child_conn,label=f'vehicle{i}')
            _ = webrtc_client.run_client_in_new_process(send_pipe=recv_fa_conn)
        # wait for some time of the establish of data channel, maybe several seconds...
        time.sleep(5)
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
            threads = []
            for i, single_cav in enumerate(single_cav_list):
                single_cav.update_info()
                control = single_cav.run_step()
                # print(len(pickle.dumps(control)))
                data_dict = control_to_dict(control)
                # print(data_dict)
                parent_conn[f'vehicle{i}'].send(pickle.dumps(data_dict))
                _ = threading.Thread(target=recv_control, args=(recv_ch_conn[f'vehicle{i}'],single_cav))
                threads.append(_)
                _.start()
                # data_ctrl = dict_to_control(data_dict)
                # single_cav.vehicle.apply_control(data_ctrl)
            for thread in threads:
                thread.join()

    finally:
        eval_manager.evaluate()

        if opt.record:
            scenario_manager.client.stop_recorder()

        scenario_manager.close()

        for v in single_cav_list:
            v.destroy()
        for v in bg_veh_list:
            v.destroy()

