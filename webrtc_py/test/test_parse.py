import argparse


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # add arguments to the parser
    parser.add_argument('--webrtc', action='store_true', help='if use webrtc_server and webrtc_client')
    # parse the arguments and return the result
    opt = parser.parse_args()
    return opt


def main():
    # parse the arguments
    opt = arg_parse()
    print(opt.webrtc)


if __name__ == '__main__':
    main()