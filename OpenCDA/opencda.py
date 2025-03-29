# -*- coding: utf-8 -*-
"""
Script to run different scenarios.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import importlib
import os
import sys
import time
import pickle
import asyncio
import threading
from omegaconf import OmegaConf

from opencda.version import __version__


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # add arguments to the parser
    parser.add_argument('-t', "--test_scenario", required=True, type=str,
                        help='Define the name of the scenario you want to test. The given name must'
                             'match one of the testing scripts(e.g. single_2lanefree_carla) in '
                             'opencda/scenario_testing/ folder'
                             ' as well as the corresponding yaml file in opencda/scenario_testing/config_yaml.')
    parser.add_argument("--record", action='store_true',
                        help='whether to record and save the simulation process to .log file')
    parser.add_argument("--apply_ml",
                        action='store_true',
                        help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
                             'Set it to true only when you have installed the pytorch/sklearn package.')
    parser.add_argument('-v', "--version", type=str, default='0.9.11',
                        help='Specify the CARLA simulator version, default'
                             'is 0.9.11, 0.9.12 is also supported.')
    parser.add_argument('--webrtc', action='store_true', help='if use webrtc_server and webrtc_client')
    # parse the arguments and return the result
    opt = parser.parse_args()
    return opt


async def run_with_webrtc(webrtc_server, webrtc_client):
    asyncio.create_task(webrtc_server.run())
    await asyncio.sleep(3)
    await webrtc_client.start()


def run_server(webrtc_server, server_loop):
    # server_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(server_loop)
    server_loop.run_until_complete(webrtc_server.run())

def run_client(webrtc_client, client_loop):
    client_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(client_loop)
    client_loop.run_until_complete(webrtc_client.start())

async def run_test(webrtc_server):
    # asyncio.set_event_loop(loop)
    await asyncio.create_task(webrtc_server.add_video_track(len(webrtc_server.video_tracks), source='video_file', file_path='/home/bupt/cykkk/carla&opencda/webrtc_py/exam_video/test1.mp4'))
    await asyncio.create_task(webrtc_server.add_data_channel('test1'))
    print(id(webrtc_server.data_channels['test1']))
    count = 0
    while True:
        await asyncio.sleep(1)
        count += 1
        webrtc_server.data_channels['test1'].send(pickle.dumps(f'hello {count}'))
        if count > 100:
            break

def test(webrtc_server, loop):
    future = asyncio.run_coroutine_threadsafe(run_test(webrtc_server), loop)
    future.result()  # 等待完成
    # asyncio.run(run_test(webrtc_server, loop))

def run_scene(scenario_runner, opt, scene_dict, webrtc_server, webrtc_client, server_loop, client_loop):
    try:
        scenario_runner(opt, scene_dict, webrtc_server=webrtc_server, webrtc_client=webrtc_client, server_loop=server_loop, client_loop=client_loop)
    except:
        return
    



def main():
    # parse the arguments
    opt = arg_parse()
    # print the version of OpenCDA
    print("OpenCDA Version: %s" % __version__)
    # set the default yaml file
    default_yaml = config_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/default.yaml')
    # set the yaml file for the specific testing scenario
    config_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'opencda/scenario_testing/config_yaml/%s.yaml' % opt.test_scenario)
    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = OmegaConf.load(default_yaml)
    scene_dict = OmegaConf.load(config_yaml)
    # merge the dictionaries
    scene_dict = OmegaConf.merge(default_dict, scene_dict)

    # import the testing script
    testing_scenario = importlib.import_module(
        "opencda.scenario_testing.%s" % opt.test_scenario)
    # check if the yaml file for the specific testing scenario exists
    if not os.path.isfile(config_yaml):
        sys.exit(
            "opencda/scenario_testing/config_yaml/%s.yaml not found!" % opt.test_cenario)

    # get the function for running the scenario from the testing script
    scenario_runner = getattr(testing_scenario, 'run_scenario')
    # run the scenario testing
    if opt.webrtc:
        sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'webrtc_py'))
        # 动态导入模块
        sender_module = importlib.import_module('src.sender_class')
        receiver_module = importlib.import_module('src.receiver_class')
        # 获取类
        Webrtc_server = getattr(sender_module, 'Webrtc_server')
        Webrtc_client = getattr(receiver_module, 'Webrtc_client')
        webrtc_server = Webrtc_server('127.0.0.1', 8080)
        webrtc_client = Webrtc_client('127.0.0.1', 8080)
        # task1 = asyncio.create_task(run_with_webrtc(webrtc_server, webrtc_client))
        # await asyncio.sleep(6)
        # run_scene(scenario_runner, opt, scene_dict, webrtc_server=webrtc_server, webrtc_client=webrtc_client)
        # await task1
        server_loop = asyncio.new_event_loop()
        client_loop = None
        # client_loop = asyncio.new_event_loop()
        thread1 = threading.Thread(target=run_server, args=(webrtc_server,server_loop))
        thread2 = threading.Thread(target=run_client, args=(webrtc_client,client_loop))
        # thread4 = threading.Thread(target=test, args=(webrtc_server,loop))
        thread3 = threading.Thread(target=run_scene, args=(scenario_runner, opt, scene_dict, webrtc_server, webrtc_client, server_loop, client_loop))

        thread1.start()
        time.sleep(3)
        thread2.start()
        time.sleep(3)
        # thread4.start()
        thread3.start()
        thread1.join()
        # task1 = asyncio.create_task(run_webrtc(webrtc_server, webrtc_client))
        # asyncio.run(run_webrtc(webrtc_server, webrtc_client))
    else:
        scenario_runner(opt, scene_dict)


if __name__ == '__main__':
    try:
        main()
        # asyncio.run(main())
    except KeyboardInterrupt:
        print(' - Exited by user.')
