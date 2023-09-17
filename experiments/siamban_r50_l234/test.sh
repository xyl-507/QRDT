export PYTHONPATH=/home/xyl/xyl-code/siamban-DROL:$PYTHONPATH

START=11
END=20
for i in $(seq $START $END)
do
    python -u ../../tools/test.py \
        --snapshot /home/xyl/xyl-code/siamban-DROL/experiments/siamban_r50_l234/xyl/15.siamban+template+CAM/snapshot/checkpoint_e$i.pth \
	      --config config-cam.yaml \
	      --gpu_id $(($i % 2)) \
	      --skip \
	      --dataset HOB 2>&1 | tee logs/test_dataset.log &
done
# python ../../tools/eval.py --tracker_path ./results --dataset UAV123 --num 4 --tracker_prefix 'ch*'

# learn from siammask
