# apartment_verification
GIT_SSH_COMMAND='ssh -i /home/model/users/ff/.ssh/id_rsa' git push

# setup

    git clone git@github.com:finfando/indoor_scene_identification.git --recurse-submodules
    pip install -r requirements.txt
    pip install -e .

# Tasks

## Real estate dataset

### generate_meta

    python indoorhack/tasks/generate_meta.py --dataset_type=real_estate --dataset_name=sonar

### generate_representations

    python indoorhack/tasks/generate_representations.py --dataset_type=real_estate --dataset_name=sonar --model_type=hash
    python indoorhack/tasks/generate_representations.py --dataset_type=real_estate --dataset_name=sonar --model_type=orb
    python indoorhack/tasks/generate_representations.py --dataset_type=real_estate --dataset_name=sonar --model_type=netvlad
    python indoorhack/tasks/generate_representations.py --dataset_type=real_estate --dataset_name=sonar --model_type=facenet

### generate_pairs

    python indoorhack/tasks/generate_pairs.py --dataset_type=real_estate --dataset_name=sonar

### get_distances

    python indoorhack/tasks/get_distances.py --dataset_type=real_estate --dataset_name=sonar --model_type=hash
    python indoorhack/tasks/get_distances.py --dataset_type=real_estate --dataset_name=sonar --model_type=orb
    python indoorhack/tasks/get_distances.py --dataset_type=real_estate --dataset_name=sonar --model_type=netvlad
    python indoorhack/tasks/get_distances.py --dataset_type=real_estate --dataset_name=sonar --model_type=facenet

## train

    python indoorhack/tasks/train.py --experiment_name=indoorhackv5 --model_type=indoorhack --epochs=1000 --stdev=20 --lr=0.01
