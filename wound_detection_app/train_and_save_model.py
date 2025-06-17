import os
from tensorflow import keras
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # fallback for older TF, can be removed if not needed
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

def train_and_save():
    dataset_path = 'dataset'
    
    # If dataset folder contains just one zip-extracted folder
    folders = os.listdir(dataset_path)
    if len(folders) == 1:
        dataset_path = os.path.join(dataset_path, folders[0])

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_data = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),  # More detail for MobileNetV2
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

    val_data = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    # Load MobileNetV2 base
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze base layers

    # Add custom top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(len(train_data.class_indices), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train model
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=30,
        callbacks=[early_stop]
    )

    # Save model
    os.makedirs('model', exist_ok=True)
    model.save('model/wound_model.h5')

    print("✅ Model trained and saved successfully.")
