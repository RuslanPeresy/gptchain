import os
import runpod
from dotenv import load_dotenv

from deploy.config import POD_CONF, GPU_COUNT

load_dotenv()

runpod.api_key = os.getenv('RUNPOD_API_KEY')


def deploy_llm(model_id=None):
    pod_conf = POD_CONF.copy()
    if model_id:
        pod_conf.update({
            "docker_args": f"--model-id {model_id} --num-shard {GPU_COUNT}",
        })
    pod = runpod.create_pod(**pod_conf)
    return f'https://{pod["id"]}-80.proxy.runpod.net'
