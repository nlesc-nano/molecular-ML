import wandb
import tensorflow as tf
import IPython
from matplotlib import pyplot as plt
import numpy as np

from data_loader import load_data
from model_loader import load_model_and_tokenizer


def get_best_sweep(entity_project_sweep_id):
    """
    Gets the run with the lowest val_root_mean_squared_error.
    Downloads model_best.h5 to .
    Unfortunately the run id is not good for loading the model
    (I had to put the id manually because the folders have also a timestamp)
    """
    api = wandb.Api()
    sweep = api.sweep(entity_project_sweep_id)
    runs = sorted(sweep.runs,
    key=lambda run: run.summary.get("val_root_mean_squared_error", 0), reverse=False)
    val_rmse = runs[0].summary.get("val_root_mean_squared_error", 0)
    print(f"Best run {runs[0].name} with {val_rmse}% validation root_mean_squared_error")
    return str(runs[0].id)

def restore_model_wandb(best_run_path, model):
    """
    Restores the best model of a sweep.
    Args:
     best_run_path: str,
     model: The original model, in our case cehemberta
    Returns:
     The model with the weights found during the best training run
    """
    best_model = wandb.restore('model-best.h5', run_path=best_run_path, replace=True)
    model.load_weights(best_model.name)
    return model

def plot_model_evaluation(model, test_dataset):
    """
    Plots the predictions and labels of the test dataset. 
    This is just because Luisa likes to visualize the results.
    """
    test_predictions=model.predict(test_dataset)
    test_labels=tf.concat([y for x, y in test_dataset], axis=0)
    plt.figure(100)
    plt.plot(test_predictions.logits[:100],'r.',fillstyle='none',label="prediction")
    plt.plot(test_labels[:100],'b.',fillstyle='none',label="truth")
    plt.legend(); plt.ylabel("cone_angle/180")
    plt.savefig("bestModel_eval.pdf")

    plt.plot(test_predictions.logits[:100])

def plot_predictions(datasets, best_model, path):
    true_angles = tf.concat([y for X, y in datasets['val']], axis=0)
    predictions = best_model.predict(datasets['val'])
    predicted_angles = tf.reshape(tf.convert_to_tensor(predictions['logits']), true_angles.shape)
    errors =  predicted_angles - true_angles

    print('test root mse: ', np.sqrt(np.mean(errors**2)))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_xlabel('true angle / 180')
    ax2.set_xlabel('true angle / 180')
    ax1.set_ylabel('predicted - true')
    ax2.set_ylabel('count')
    ax1.scatter(true_angles, errors)
    ax2.hist(predicted_angles, bins=100)
    fig.savefig(path)

def plot_filtered_stats(datasets, best_model, path):
    """
    Use the predicted angles to select the fraction x of the validation data with
    the lowest predicted angle.
    Then measure statistics of the true angles for that selection,
    and plot as a function of the fraction.
    """
    true_angles = tf.concat([y for X, y in datasets['val']], axis=0)
    predictions = best_model.predict(datasets['val'])
    predicted_angles = tf.reshape(tf.convert_to_tensor(predictions['logits']), true_angles.shape)

    sorted_indices = tf.argsort(predicted_angles)
    sorted_predictions = tf.gather(predicted_angles, sorted_indices)
    sorted_truths = tf.gather(true_angles, sorted_indices)

    percentages = tf.range(len(predicted_angles)) / len(predicted_angles)
    percentages = percentages[:(len(percentages) // 5)]
    means, maxes = [], []
    # for some reason doing this in a list comprehension didn't work
    for i in range(1, len(percentages) + 1):
        means.append(tf.reduce_mean(sorted_truths[:i]))
        maxes.append(tf.reduce_max(sorted_truths[:i]))
    means = tf.concat(means, axis=0)
    maxes = tf.concat(maxes, axis=0)

    fig, ax = plt.subplots()
    ax.set_xlabel('Fraction Kept')
    ax.plot(percentages, means)
    ax.plot(percentages, maxes)
    ax.legend(['mean', 'max'])

    fig.savefig(path)

if __name__ == '__main__':
    #model loading
    model_orig, tokenizer = load_model_and_tokenizer('DeepChem/ChemBERTa-77M-MTR')
    entity_project_sweep_id ="luisaforozco/chemberta/z1e2ggux/"
    entity_project_sweep_id ="apjansen/chemberta/5ble16ck/"
    entity_project_sweep_id = 'apjansen/chemberta/1pzw7wz8/'
    best_run_id=get_best_sweep(entity_project_sweep_id)
    best_model=restore_model_wandb(entity_project_sweep_id+best_run_id,model_orig)

    #model evaluation
    datasets = load_data(
        filename='../data/cone_angle_carbox_11K.csv',
        tokenizer=tokenizer,
        task_id=0,
        batch_size=512,
    )

    best_model.compile(
        optimizer='Adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    results = best_model.evaluate(datasets['val'], batch_size=32)

    plot_predictions(datasets, best_model, path='prediction_errors.jpg')
    plot_filtered_stats(datasets, best_model, path='filtered_stats.jpg')

    print("test loss, test root mse:", results)
    plot_model_evaluation(best_model,datasets['val'])
