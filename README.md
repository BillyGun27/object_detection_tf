still trying to run without set PYTHONPATH=D:\FinalProject\ComputerVision\models\research\slim
test python builders/model_builder_test.py
reference from google drive Setup Object Detection Python Tensorflow

# Setup
Protobuf Compilation (download and run protoc.exe)
    protoc object_detection/protos/*.proto --python_out=.

Pycocotools
    pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI

# Important file for training 
    1. Tfrecord (create from dataset)
    2. config (example from samples/configs)
    3. pbtxt (example from data)


# Prepare Dataset 
## Create TFRecord
    1. Prepare voc format xml and image 
    2. run xml_to_csv.py 
    3. run generate_tfrecord.py
        python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record --image_dir=macaroni_dataset

        python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record --image_dir=macaroni_dataset

# Training
    
    python model_main.py --pipeline_config_path="data_setup\ssd_mobilenet_v1_pets.config" --model_dir="log/model" --num_train_steps=200 --sample_1_of_n_eval_examples=1 --alsologtostderr