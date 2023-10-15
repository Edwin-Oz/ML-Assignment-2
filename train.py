from keras.preprocessing.image import ImageDataGenerator
from model import build_model
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Print class indices
print(train_generator.class_indices)

model = build_model((128, 128, 3))

history = model.fit(train_generator, epochs=10)
model.save('hotdog_classifier.h5')
