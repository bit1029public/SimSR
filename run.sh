
DOMAIN=cartpole
TASK=swingup_sparse
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
