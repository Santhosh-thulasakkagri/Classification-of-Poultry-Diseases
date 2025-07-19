from model import build_vgg16_model, train_model
from utils import plot_training_history, evaluate_model, save_model
from data import create_data_generators

def main():
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Build and train model
    model = build_vgg16_model()
    history = train_model(model, train_gen, val_gen, epochs=20)
    
    # Evaluate and save results
    plot_training_history(history)
    evaluate_model(model, test_gen)
    save_model(model, 'poultry_disease_vgg16')
    
    # Hyperparameter tuning (optional)
    # from model import hyperparameter_tuning
    # tuned_model = hyperparameter_tuning(train_gen, val_gen)
    # history_tuned = train_model(tuned_model, train_gen, val_gen)

if __name__ == "__main__":
    main()
