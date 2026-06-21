import tensorflow as tf
from main import create_full_model  # because create_full_model is in your main file

def train_model(data_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='categorical',
        subset='training'
    )
    
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(128, 128),
        color_mode='grayscale',
        class_mode='categorical',
        subset='validation'
    )
    
    model = create_full_model()
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=50
    )
    
    model.save('blood_group_model.h5')
    print("Model saved as blood_group_model.h5")
    return history

if __name__ == "__main__":
    train_model("dataset")  # Replace "dataset" with your dataset folder name
