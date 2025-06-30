import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import kerastuner as kt

# Constants
IMG_SIZE = (224, 224)
NUM_CLASSES = 4

def build_vgg16_model():
    """Build and compile VGG16 model"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classifier
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_vgg19_model():
    """Build and compile VGG19 model"""
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_resnet50_model():
    """Build and compile ResNet50 model"""
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def model_builder(hp):
    """Hyperparameter tuning model builder"""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = Flatten()(x)
    
    # Tune number of units in Dense layer
    hp_units = hp.Int('units', min_value=128, max_value=512, step=128)
    x = Dense(units=hp_units, activation='relu')(x)
    
    # Tune dropout rate
    if hp.Boolean('dropout'):
        x = tf.keras.layers.Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)
    
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Tune learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def hyperparameter_tuning(train_gen, val_gen):
    """Perform hyperparameter tuning with Hyperband"""
    tuner = kt.Hyperband(
        model_builder,
        objective='val_accuracy',
        max_epochs=10,
        factor=3,
        directory='hyperband_tuning',
        project_name='poultry_disease'
    )
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    
    tuner.search(
        train_gen,
        validation_data=val_gen,
        epochs=50,
        callbacks=[early_stop]
    )
    
    # Get optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Optimal units: {best_hps.get('units')}")
    print(f"Optimal learning rate: {best_hps.get('learning_rate')}")
    
    # Build model with optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)
    return model

def train_model(model, train_gen, val_gen, epochs=10):
    """Train model with early stopping"""
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early_stop]
    )
    return history
