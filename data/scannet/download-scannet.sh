VIRTUALENV_PATH=/home/model/.virtualenvs/filip/bin/activate
SCENES_LIST_PATH=../submodules/ScanNet/Tasks/Benchmark/scannetv2_train.txt
OUTPUT_PATH=/home/model/users/ff/scannet_data/scannet_train

source $VIRTUALENV_PATH
while IFS= read -r line <&9; do
    echo "Downloading .sens file for scene $line"
    python download-scannet.py --id $line --type .sens -o $OUTPUT_PATH
    echo "Reading .sens file for scene $line"
    python ../submodules/ScanNet/SensReader/python/reader.py --filename $OUTPUT_PATH/scans/$line/$line.sens --export_color_images --output_path $OUTPUT_PATH/images/$line
    echo "Removing .sens file for scene $line"
    rm -rf $OUTPUT_PATH/scans/$line
done 9< "$SCENES_LIST_PATH"
