import logging
from pathlib import Path
import tensorflow as tf
import time
import mlflow
import mlflow.keras
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_next_experiment_name(base_name, base_path):
    n = 0
    while True:
        experiment_name = f"{base_name}_{n}"
        if not (base_path / f"{experiment_name}.h5").exists():
            break
        n += 1
    return experiment_name

def get_base_model(base_model_path, image_size, weights, include_top):
    logger.info(">>>>>>>>>>>>>>>> Loading the base model... <<<<<<<<<<<<<<<<<<<<<<<<<<")
    model = tf.keras.applications.vgg16.VGG16(
        input_shape=image_size,
        weights=weights,
        include_top=include_top
    )
    save_model(base_model_path, model)
    logger.info(">>>>>>>>>>>>>>> Base model loaded and saved. <<<<<<<<<<<<<<<<<<")
    return model

def prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
    if freeze_all:
        for layer in model.layers:
            layer.trainable = False
    elif (freeze_till is not None) and (freeze_till > 0):
        for layer in model.layers[:-freeze_till]:
            layer.trainable = False

    flatten_in = tf.keras.layers.Flatten()(model.output)
    prediction = tf.keras.layers.Dense(
        units=classes,  # Adjust units to the number of classes
        activation="softmax"
    )(flatten_in)

    full_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=prediction
    )

    full_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    full_model.summary()
    return full_model

def save_model(path: Path, model: tf.keras.Model):
    path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    model.save(path)
    logger.info(f"Model saved at {path}")

def model_trainer(
        updated_base_model_path, trained_model_path, training_data,
        params_epochs, params_batch_size, params_is_augmentation,
        params_image_size, learning_rate, experiment_name):

    # Load and save base model
    base_model = get_base_model(updated_base_model_path, params_image_size, "imagenet", False)

    # Update and save the full model
    full_model = prepare_full_model(base_model, 2, freeze_all=True, freeze_till=15, learning_rate=learning_rate)

    # Data generators
    datagenerator_kwargs = dict(
        rescale=1./255,
        validation_split=0.20
    )

    dataflow_kwargs = dict(
        target_size=params_image_size[:-1],
        batch_size=params_batch_size,
        interpolation="bilinear"
    )

    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagenerator_kwargs
    )

    valid_generator = valid_datagenerator.flow_from_directory(
        directory=training_data,
        subset="validation",
        shuffle=False,
        **dataflow_kwargs
    )

    if params_is_augmentation:
        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=40,
            horizontal_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            **datagenerator_kwargs
        )
    else:
        train_datagenerator = valid_datagenerator

    train_generator = train_datagenerator.flow_from_directory(
        directory=training_data,
        subset="training",
        shuffle=True,
        **dataflow_kwargs
    )

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = valid_generator.samples // valid_generator.batch_size

    start_time = time.time()
    
    # Define a learning rate scheduler callback
    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = full_model.fit(
        train_generator,
        epochs=params_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_data=valid_generator,
        callbacks=[lr_callback]
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds.")

    # Save model
    save_model(trained_model_path / f"{experiment_name}.h5", full_model)

    return training_time, history  # Return the training time and history

def model_evaluation(path_of_model, params_image_size, params_batch_size, training_data, mlflow_uri, training_time, model_params, experiment_name):
    logger.info(">>>>>> Model Evaluation Stage started <<<<<<")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(mlflow_uri)

    # Define experiment name
    experiment_base_name = "cnn_project_experiment"
    experiment_name = experiment_base_name
    n = 0

    while mlflow.get_experiment_by_name(experiment_name):
        n += 1
        experiment_name = f"{experiment_base_name}_{n}"

    if not mlflow.get_experiment_by_name(experiment_name):
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    # Load model
    model = tf.keras.models.load_model(path_of_model)

    # Data generator for validation
    datagenerator_kwargs = dict(
        rescale=1./255,
        validation_split=0.20
    )

    dataflow_kwargs = dict(
        target_size=params_image_size[:-1],
        batch_size=params_batch_size,
        interpolation="bilinear"
    )

    valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
        **datagenerator_kwargs
    )

    valid_generator = valid_datagenerator.flow_from_directory(
        directory=training_data,
        subset="validation",
        shuffle=False,
        **dataflow_kwargs
    )

    # Evaluate model
    y_true = valid_generator.classes
    y_pred = np.argmax(model.predict(valid_generator), axis=-1)

    score = model.evaluate(valid_generator)
    logger.info(f"Validation Loss: {score[0]}")
    logger.info(f"Validation Accuracy: {score[1]}")

    # Compute additional metrics
    precision = classification_report(y_true, y_pred, output_dict=True)['weighted avg']['precision']
    recall = classification_report(y_true, y_pred, output_dict=True)['weighted avg']['recall']
    f1_score = classification_report(y_true, y_pred, output_dict=True)['weighted avg']['f1-score']
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Log metrics and model to MLflow
    with mlflow.start_run(experiment_id=experiment_id):

        mlflow.log_metrics({
            "loss": score[0],
            "accuracy": score[1],
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        })
        mlflow.log_metric("training_time", training_time)
        mlflow.log_params(model_params)

        # Save and log confusion matrix as an artifact
        conf_matrix_path = f"{experiment_name}_confusion_matrix.json"
        with open(conf_matrix_path, 'w') as f:
            json.dump(conf_matrix.tolist(), f)
        mlflow.log_artifact(conf_matrix_path)

        # Optionally, save confusion matrix as an image
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        conf_matrix_img_path = f"{experiment_name}_confusion_matrix.png"
        plt.savefig(conf_matrix_img_path)
        mlflow.log_artifact(conf_matrix_img_path)

        # Log model
        mlflow.keras.log_model(model, "model")

    logger.info(f"Confusion matrix saved to {conf_matrix_path} and {conf_matrix_img_path}")

def main():
    # Configuration setup
    root_dir = Path(__file__).parent
    models_dir = root_dir / "models"
    trained_model_path = models_dir

    training_data = root_dir / "dataset"
    params_epochs = 20
    params_batch_size = 64
    params_is_augmentation = True
    params_image_size = [224, 224, 3]
    mlflow_uri = "http://127.0.0.1:5000"
    learning_rate = 0.0001

    # Generate a unique experiment name
    experiment_base_name = "experiment"
    experiment_name = get_next_experiment_name(experiment_base_name, models_dir)

    model_params = {
        "epochs": params_epochs,
        "batch_size": params_batch_size,
        "is_augmentation": params_is_augmentation,
        "learning_rate": learning_rate,
        "experiment_name": experiment_name
    }

    try:
        logger.info(">>>>>> Model Training Stage started <<<<<<")
        
        training_time, history = model_trainer(
            models_dir / f"{experiment_name}_base_model.h5", trained_model_path, training_data,
            params_epochs, params_batch_size, params_is_augmentation,
            params_image_size, learning_rate, experiment_name
        )

        logger.info(">>>>>> Model Training Stage completed <<<<<<")

        logger.info(">>>>>> Model Evaluation Stage started <<<<<<")
        
        model_evaluation(
            trained_model_path / f"{experiment_name}.h5", params_image_size, params_batch_size, training_data, mlflow_uri, training_time, model_params, experiment_name
        )
        
        logger.info(">>>>>> Model Evaluation Stage completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == '__main__':
    main()
