all:
	python3 main.py --save --save-path models --num-agents 2 --layout large_overcooked_layout  --centralised --batch-size 128 --num-minibatches 4 \
	--total-steps 1000000

cramped:
	python3 main.py --save-path models --num-agents 1 --num-envs 1 --layout overcooked_cramped_room_v0  --num-steps 256 --num-minibatches 4 \
	--total-steps 20000000 --seed 3 --log --ppo-epoch 5 --clip-param 0.05 \
	--value-loss-coef 0.1 --entropy-coef 0.01 --gamma 0.99 --lam 0.95 --max-grad-norm 0.5 --lr 3e-4 --data-path data \
	--feature local_obs

forced:
	python3 main.py --save-path models --num-agents 2 --num-envs 16 --layout overcooked_forced_coordination_v0  --num-steps 256 --num-minibatches 4 \
	--total-steps 20000000 --seed 3 --log --ppo-epoch 5 --clip-param 0.05 \
	--value-loss-coef 0.1 --entropy-coef 0.01 --gamma 0.99 --lam 0.95 --max-grad-norm 0.5 --lr 3e-4 --data-path data0721_forced_coord_minimal_spatial \
	--feature global_obs 

debug:
	CUDA_LAUNCH_BLOCKING=1 python3 main.py --save-path models --num-agents 2 --layout overcooked_cramped_room_v0  --centralised --batch-size 1000 --num-minibatches 5 \
		--total-steps 100000 --log --seed 2

# QMIX targets
qmix-cramped:
	python3 main.py --algorithm qmix --save-path models --num-agents 2 --num-envs 1 --layout overcooked_cramped_room_v0 \
	--num-episodes 5000 --seed 1 --lr 5e-4 --gamma 0.99 --data-path data \
	--epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.70 --target-update-freq 200 \
	--buffer-size 1000000 --batch-size-qmix 32 --mixing-embed-dim 32 --hidden-dim 256 \
	--feature global_obs --save --log

qmix-forced:
	python3 main.py --algorithm qmix --save-path models --num-agents 2 --num-envs 1 --layout overcooked_forced_coordination_v0 \
	--num-episodes 5000 --seed 1 --lr 5e-4 --gamma 0.99 --data-path data \
	--epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.995 --target-update-freq 200 \
	--buffer-size 10000 --batch-size-qmix 32 --mixing-embed-dim 32 --hidden-dim 256 \
	--feature Minimal_spatial --save --log

qmix-large:
	python3 main.py --algorithm qmix --save-path models --num-agents 4 --num-envs 1 --layout large_overcooked_layout \
	--num-episodes 8000 --seed 1 --lr 5e-4 --gamma 0.99 --data-path data \
	--epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay 0.9995 --target-update-freq 300 \
	--buffer-size 15000 --batch-size-qmix 64 --mixing-embed-dim 64 --hidden-dim 512 \
	--feature global_obs --save --log

qmix-debug:
	CUDA_LAUNCH_BLOCKING=1 python3 main.py --algorithm qmix --save-path models --num-agents 2 --num-envs 1 \
	--layout overcooked_cramped_room_v0 --num-episodes 100 --seed 1 --lr 1e-3 --gamma 0.99 \
	--epsilon-start 1.0 --epsilon-end 0.1 --epsilon-decay 0.99 --target-update-freq 50 \
	--buffer-size 1000 --batch-size-qmix 16 --mixing-embed-dim 16 --hidden-dim 128 \
	--feature global_obs --log

sarsa:
	python3 main.py --algorithm sarsa --save-path models --num-agents 1 --num-envs 1 --layout overcooked_cramped_room_v0 \
	--total-steps 1000000 --seed 1 --step-size 0.01 --data-path data \
	--feature local_obs

# Test QMIX implementation
test-qmix:
	python3 tests/test_qmix.py

# Integration test with actual environment
test-qmix-integration:
	python3 tests/test_qmix_integration.py

help:
	@echo "Available targets:"
	@echo "  all          - Default MAPPO run with centralized critic"
	@echo "  cramped      - MAPPO on cramped room layout"
	@echo "  forced       - MAPPO on forced coordination layout"
	@echo "  debug        - Debug run with CUDA blocking"
	@echo "  qmix-cramped - QMIX on cramped room layout"
	@echo "  qmix-forced  - QMIX on forced coordination layout"
	@echo "  qmix-large   - QMIX on large layout with 4 agents"
	@echo "  qmix-debug   - Debug QMIX run"
	@echo "  test-qmix    - Run QMIX unit tests"
	@echo "  test-qmix-integration - Run QMIX integration test with environment"
	@echo "  help         - Show this help message"