# SimSR
Code and dataset for the paper [SimSR: Simple Distance-based State Representationfor Deep Reinforcement Learning]() (AAAI-22).

## Requirements
We assume you have access to a gpu that can run CUDA 11. All of the dependencies are in the `conda_env.yml` file.

```
conda env create -f conda_env.yml
```

After the instalation ends you can activate your environment with

```
conda activate simsr
```

## Instructions

To train a SimSR agent on the `cartpole swingup` task from image-based observations run `bash run.sh` from the root of this directory. The `run.sh` file contains the following command, which you can modify to try different environments / hyperparamters.

```
DOMAIN=cartpole
TASK=swingup
SEED=1

MUJOCO_GL="egl" CUDA_VISIBLE_DEVICES=0 nohup python -u train.py \
	--domain_name ${DOMAIN} \
	--task_name ${TASK} \
	--encoder_type pixel \
	--action_repeat 4 \
	--pre_transform_image_size 84 \
	--image_size 84 \
	--work_dir ./tmp \
	--agent simsr_sac \
	--frame_stack 3\
	--seed ${SEED} --critic_lr 1e-3 \
	--actor_lr 1e-3 \
	--eval_freq 10000 \
	--batch_size 128 \
	--num_train_steps 260000 > ${DOMAIN}_${TASK}_${SEED}.log &
```

Note that the MuJoCo Python bindings support three different OpenGL rendering backends: `"glfw"`, `"egl"`, or `"osmesa"`. You can also specify a particular backend to use by setting the `MUJOCO_GL=` environment variable to one of them.

To visualize progress with tensorboard run:

```
tensorboard --logdir ./path/to/your/log --port 6006
```

## References
Please cite the paper [SimSR: Simple Distance-based State Representationfor Deep Reinforcement Learning]() if you found the resources in the repository useful.

```

```
