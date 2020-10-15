dvc run \
    --no-exec --name=indoorhack-v10 --force \
    -o "./indoorhack-v5-10.torch" \
    -o "./indoorhack-v5-10.done" \
    "python ../indoorhack/tasks/train.py --experiment_name=indoorhack-v5-10 --model_type=indoorhack --epochs=1000 --stdev=10 --lr=.0001"

dvc run \
    --no-exec --name=indoorhack-v10 --force \
    -o "./indoorhack-v5-20.torch" \
    -o "./indoorhack-v5-20.done" \
    "python ../indoorhack/tasks/train.py --experiment_name=indoorhack-v5-20 --model_type=indoorhack --epochs=1000 --stdev=20 --lr=.0001"

dvc run \
    --no-exec --name=indoorhack-v10 --force \
    -o "./indoorhack-v5-30.torch" \
    -o "./indoorhack-v5-30.done" \
    "python ../indoorhack/tasks/train.py --experiment_name=indoorhack-v5-30 --model_type=indoorhack --epochs=1000 --stdev=30 --lr=.0001"
