all:
	python3 main.py --save --save-path models --num-agents 2 --layout large_overcooked_layout  --centralised --batch-size 128 --num-minibatches 4 \
	--total-steps 1000000

cramped:
	python3 main.py --save-path models --num-agents 2 --layout overcooked_cramped_room_v0  --batch-size 128 --num-minibatches 4 \
	--total-steps 1000000 --seed 2 --log --centralised --ppo-epoch 10 --clip-param 0.2 \
	--value-loss-coef 0.5 --entropy-coef 0.01 --gamma 0.99 --lam 0.95 --max-grad-norm 0.5 --lr 1e-4

debug:
	CUDA_LAUNCH_BLOCKING=1 python3 main.py --save-path models --num-agents 2 --layout overcooked_cramped_room_v0  --centralised --batch-size 1000 --num-minibatches 5 \
		--total-steps 100000 --log --seed 2