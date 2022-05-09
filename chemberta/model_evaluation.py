import wandb
from data_loader import load_data
from model_loader import load_model_and_tokenizer
import tensorflow as tf
import IPython
from matplotlib import pyplot as plt

def get_best_sweep(entity_project_sweep_id):
    """
    Gets the run with the lowest val_root_mean_squared_error.
    Downloads model_best.h5 to ./ 
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

def restore_model_wandb(best_run_path,model):
    """
    Restores the best model of a sweep. 
    Args:
     best_run_path: str, 
     model: The original model, in our case cehemberta
    Returns:
     The model with the weights found during the best training run
    """
    best_model = wandb.restore('model-best.h5', run_path=best_run_path)
    model.load_weights(best_model.name)
    return model

def plot_model_evaluation(model,test_dataset):
    """
    Plots the predictions and labels of the test dataset. 
    This is just because Luisa likes to visualize the results.
    """
    test_predictions=model.predict(test_dataset)
    test_labels=tf.concat([y for x, y in test_dataset], axis=0)
    #IPython.embed()
    plt.figure(100)
    plt.plot(test_predictions.logits[:100],'r.',fillstyle='none',label="prediction")
    plt.plot(test_labels[:100],'b.',fillstyle='none',label="truth")
    plt.legend(); plt.ylabel("cone_angle/180")
    plt.savefig("bestModel_eval.pdf")
    plt.figure(200)
    plt.plot(test_predictions.logits[:100],test_labels[:100],'k.')
    plt.ylabel("Predictied cone_angle/180"); plt.xlabel("Truth cone_angle/180")
    plt.savefig("bestModel_eval_2.pdf")


if __name__ == '__main__':
    #model loading
    model_orig, tokenizer = load_model_and_tokenizer('DeepChem/ChemBERTa-77M-MTR')
    entity_project_sweep_id="luisaforozco/chemberta/z1e2ggux/"
    best_run_id=get_best_sweep(entity_project_sweep_id)
    best_model=restore_model_wandb(entity_project_sweep_id+best_run_id,model_orig)

    #model evaluation
    datasets = load_data(
        filename='../data/cone_angle_carbox_11K.csv',
        tokenizer=tokenizer,
        task_id=0,
        batch_size=32,
    )

    best_model.compile(
        optimizer='Adam',
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    
    results = best_model.evaluate(datasets['test'], batch_size=32)
    print("test loss, test root mse:", results)
    plot_model_evaluation(best_model,datasets['test'])