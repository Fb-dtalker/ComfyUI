import asyncio
import itertools
import os
import shutil
import threading
import gc
import time

from comfy.cli_args import args
import comfy.utils

if os.name == "nt":
    import logging
    logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

if __name__ == "__main__":
    if args.dont_upcast_attention:
        print("disabling upcasting of attention")
        os.environ['ATTN_PRECISION'] = "fp16"

    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        print("Set cuda device to:", args.cuda_device)


import yaml

import execution
import folder_paths
import server as serverPackage
from server import BinaryEventTypes
from nodes import init_custom_nodes
import comfy.model_management

def prompt_worker(q, server):
    e = execution.PromptExecutor(server)
    while True:
        item, item_id = q.get()
        execution_start_time = time.perf_counter()
        prompt_id = item[1]
        e.execute(item[2], prompt_id, item[3], item[4])
        q.task_done(item_id, e.outputs_ui)
        if server.client_id is not None:
            server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, server.client_id)

        print("Prompt executed in {:.2f} seconds".format(time.perf_counter() - execution_start_time))
        gc.collect()
        comfy.model_management.soft_empty_cache()

def loop_prompt_worker(q, server):
    e = execution.LoopPromptExecutor(server)    
    while True:
        item, item_id = q.get()
        execution_start_time = time.perf_counter()
        prompt_id = item[1]
        extra_date = item[3]

        if extra_date["anime"] is not None :
            startIndex = extra_date["anime"]["animeStartIndex"]
            endIndex = extra_date["anime"]["animeEndIndex"]
            print("AnimeLoopStep::Total:: %d -> %d" % (startIndex, endIndex))
            index = startIndex
            flag, prompt = IncrementAnimeStartNode(item[2], 0)
            while index < endIndex : 
                if not flag :
                    break
                curExecuteStartTime = time.perf_counter()
                e.execute(prompt, prompt_id, item[3], item[4])
                flag, prompt = IncrementAnimeStartNode(prompt, 1)
                print("AnimeLoopStep:::(Cur::%d)---(From::%d -> To::%d)---(Remain::%d)" % (index, startIndex, endIndex, endIndex - index))
                print("AnimeLoopStep:::(UseTime:: {:.2f} seconds)".format(time.perf_counter() - curExecuteStartTime))
                index = index + 1
            else:
                e.execute(item[2], prompt_id, item[3], item[4])
        q.task_done(item_id, e.outputs_ui)
        if server.client_id is not None:
            server.send_sync("executing", { "node": None, "prompt_id": prompt_id }, server.client_id)

        print("Prompt executed in {:.2f} seconds".format(time.perf_counter() - execution_start_time))
        gc.collect()
        comfy.model_management.soft_empty_cache()

def IncrementAnimeStartNode(origion_prompt, add_value):
    prompt = origion_prompt.copy()
    flag = False
    for k in prompt.keys():
        v = prompt[k]
        if v['class_type'] is not None and v['class_type'].endswith("_animeStepIndex") :
            v['inputs']['anime_step'] = v['inputs']['anime_step'] + add_value
            flag = True
    return flag, prompt

async def run(server, address='', port=8188, verbose=True, call_on_start=None):
    await asyncio.gather(server.start(address, port, verbose, call_on_start), server.publish_loop())


def hijack_progress(server):
    def hook(value, total, preview_image_bytes):
        server.send_sync("progress", {"value": value, "max": total}, server.client_id)
        if preview_image_bytes is not None:
            server.send_sync(BinaryEventTypes.PREVIEW_IMAGE, preview_image_bytes, server.client_id)
    comfy.utils.set_progress_bar_global_hook(hook)


def cleanup_temp():
    temp_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "temp")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)


def load_extra_path_config(yaml_path):
    with open(yaml_path, 'r') as stream:
        config = yaml.safe_load(stream)
    for c in config:
        conf = config[c]
        if conf is None:
            continue
        base_path = None
        if "base_path" in conf:
            base_path = conf.pop("base_path")
        for x in conf:
            for y in conf[x].split("\n"):
                if len(y) == 0:
                    continue
                full_path = y
                if base_path is not None:
                    full_path = os.path.join(base_path, full_path)
                print("Adding extra search path", x, full_path)
                folder_paths.add_model_folder_path(x, full_path)


if __name__ == "__main__":
    cleanup_temp()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server = serverPackage.PromptServer(loop)
    q = execution.PromptQueue(server)

    animeloop = asyncio.new_event_loop()
    asyncio.set_event_loop(animeloop)
    animeServer = serverPackage.PromptServer(animeloop)
    animeq = execution.PromptQueue(animeServer)

    extra_model_paths_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extra_model_paths.yaml")
    if os.path.isfile(extra_model_paths_config_path):
        load_extra_path_config(extra_model_paths_config_path)

    if args.extra_model_paths_config:
        for config_path in itertools.chain(*args.extra_model_paths_config):
            load_extra_path_config(config_path)

    init_custom_nodes()
    server.add_routes()
    animeServer.add_routes()
    hijack_progress(server)
    hijack_progress(animeServer)

    threading.Thread(target=prompt_worker, daemon=True, args=(q, server,)).start()
    threading.Thread(target=loop_prompt_worker, daemon=True, args=(animeq, animeServer,)).start()

    if args.output_directory:
        output_dir = os.path.abspath(args.output_directory)
        print(f"Setting output directory to: {output_dir}")
        folder_paths.set_output_directory(output_dir)

    if args.quick_test_for_ci:
        exit(0)

    call_on_start = None
    if args.auto_launch:
        def startup_server(address, port):
            import webbrowser
            webbrowser.open(f"http://{address}:{port}")
        call_on_start = startup_server

    anime_call_on_start = None
    if args.auto_launch:
        def startup_server(address, port):
            import webbrowser
            webbrowser.open(f"http://{address}:{port}")
        anime_call_on_start = startup_server

    try:
        if args.anime_mode :
            print(":::Start as Anime Mode:::")
            animeloop.run_until_complete(run(animeServer, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=anime_call_on_start))
        else:
            loop.run_until_complete(run(server, address=args.listen, port=args.port, verbose=not args.dont_print_server, call_on_start=call_on_start))
    except KeyboardInterrupt:
        print("\nStopped server")

    cleanup_temp()
