dvc run \
    --no-exec --name=indoorhack-v10-10 --force \
    -o "./indoorhack-v10-10.torch" \
    -o "./indoorhack-v10-10.done" \
    "python ../indoorhack/tasks/train.py --experiment_name=indoorhack-v10-10 --model_type=indoorhack-mobilenetv2 --epochs=1000 --stdev=10 --lr=.000001"

dvc run \
    --no-exec --name=indoorhack-v10-20 --force \
    -o "./indoorhack-v10-20.torch" \
    -o "./indoorhack-v10-20.done" \
    "python ../indoorhack/tasks/train.py --experiment_name=indoorhack-v10-20 --model_type=indoorhack-mobilenetv2 --epochs=1000 --stdev=20 --lr=.000001"

dvc run \
    --no-exec --name=indoorhack-v10-30 --force \
    -o "./indoorhack-v10-30.torch" \
    -o "./indoorhack-v10-30.done" \
    "python ../indoorhack/tasks/train.py --experiment_name=indoorhack-v10-30 --model_type=indoorhack-mobilenetv2 --epochs=1000 --stdev=30 --lr=.000001"
