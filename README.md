
### Preparing Data


- **Download Raw Data**

    You can download all datasets at [Google Drive](https://drive.google.com/drive/folders/14EJVODCU48fGK0FkyeVom_9lETh80Yjp) .
    Unzip them to `datasets/raw_data/`

- **Pre-process Data**

    ```bash
    cd /path/to/your/project
    python scripts/data_preparation/${DATASET_NAME}/generate_training_data.py
    ```

    Replace `${DATASET_NAME}` with your dataset. The processed data will be placed in `datasets/${DATASET_NAME}`.


### Train

    ```bash
    python step/run.py --cfg='model/STIM_$DATASET.py' --gpus '0'
    ```
  Replace `$DATASET_NAME` with with your dataset.

The code is developed with [BasicTS](https://github.com/zezhishao/BasicTS).

