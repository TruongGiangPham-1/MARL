all:
	python3 main.py --save --save-path models --num-agents 2 --layout large_overcooked_layout  --centralised --batch-size 128 --num-minibatches 4 \
	--total-steps 1000000

cramped:
	python3 main.py --save --save-path models --num-agents 2 --layout overcooked_cramped_room_v0  --centralised --batch-size 128 --num-minibatches 4 \
	--total-steps 1000000 --log --render