# conda activate /home/chenmouxiang/env/cmx_graph
CUDA_VISIBLE_DEVICES=3 python main.py --base configs/mug/autoencoder.yaml -t --gpus 0,  --accelerator cuda

CUDA_VISIBLE_DEVICES=3 python main.py --base configs/mug/mug_diffusion.yaml -t --gpus 0,  --accelerator cuda
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/mug/mug_diffusion.yaml -t --gpus 0,  --accelerator cuda --scale_lr False


CUDA_VISIBLE_DEVICES=3 python main.py --base configs/mug/mug_diffusion_stft.yaml -t --gpus 0,  --accelerator cuda


# extractnum -p ",,,,,,,,,,,,,{epoch},{loss},{_},{_},{_},{_},{acc_rice},{acc_ln},{precision_rice},{precision_ln},{recall_rice},{recall_ln}" --x epoch --output temp.png logs/2022-10-29T23-16-35_autoencoder/testtube/version_0/metrics.csv --placehold_pattern "[+|-]?\d+(\.\d*(e-0\d)?)?"