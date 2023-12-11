GPU_COUNT = 2

POD_CONF = {
    "name": "gptchain-0.0.2",
    "image_name": "ghcr.io/huggingface/text-generation-inference:latest",
    "gpu_type_id": "NVIDIA RTX A6000",
    "cloud_type": "SECURE",
    "docker_args": f"--model-id l3utterfly/llama2-7b-layla --num-shard {GPU_COUNT}",
    "gpu_count": GPU_COUNT,
    "volume_in_gb": 50,
    "container_disk_in_gb": 20,
    "ports": "80/http,29500/http",
    "volume_mount_path": "/data",
}
