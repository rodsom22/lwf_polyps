import run_model as rmodel


# ------------------------------- Artifact detection setup -------------------------------

def main():
    dataset_type = "--dataset-type=polyp"
    # folder where all the data is contained
    data_dir = "--data-dir=./data/"
    val_dir = "--val-dir=./data/"

    # file path to classes file
    classes = "--classes=./data/artifacts.csv"
    # Annotations for the images. It has the format image_path,x1,y1,x2,y2,class_name
    val_annotations = "--val-annotations=./data/annotations.csv"

    # folder path where our trained model is saved
    model_folder = "--model=./models/"
    # file name of trained model
    model_weights = "artefact_annotator.h5"


    # folder name of folder where annotated images will be saved
    save_folder = "results/"

    # threshold for yielding detection outputs
    threshold = "0.5"

    gpu_id = '--gpu=0'

    # backbone
    backbone = "--backbone=resnet50"
    # max detections per image
    max_detections = '--max-detections=500'
    score_threshold = "--score-threshold=" + threshold
    im_threshold = '--im-threshold=' + threshold

    ####
    mode = '--mode=detection'
    model_path = model_folder + model_weights
    # Path for storing results.
    save_path = "--save-path=./data/results/"

    arg_list = [dataset_type,
                score_threshold,
                classes,
                "--convert-model",
                backbone,
                model_path,
                max_detections,
                data_dir,
                val_dir,
                val_annotations,
                mode,
                gpu_id,
                save_path,
                im_threshold]

    test_df = rmodel.main(arg_list)
    test_df.to_csv(save_path.replace('--save-path=', '') + 'results.csv', index=False)


if __name__ == '__main__':
    main()
