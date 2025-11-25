# train.py
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 12
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)


train_gen = ImageDataGenerator(
rescale=1./255,
validation_split=0.2,
rotation_range=20,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.1,
zoom_range=0.1,
horizontal_flip=True,
fill_mode='nearest')


train_flow = train_gen.flow_from_directory(
DATASET_DIR,
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode='categorical',
subset='training')


val_flow = train_gen.flow_from_directory(
DATASET_DIR,
target_size=IMG_SIZE,
batch_size=BATCH_SIZE,
class_mode='categorical',
subset='validation')


num_classes = len(train_flow.class_indices)
print('Found classes:', train_flow.class_indices)


# Save class indices
with open(os.path.join(MODEL_DIR, 'classes.json'), 'w') as f:
json.dump(train_flow.class_indices, f)


base_model = MobileNetV2(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), weights='imagenet')
base_model.trainable = False


inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)


model.compile(optimizer=optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, 'model.h5'), monitor='val_accuracy', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


history = model.fit(train_flow, validation_data=val_flow, epochs=EPOCHS, callbacks=[checkpoint, early])


# Optionally fine-tune
base_model.trainable = True
for layer in base_model.layers[:-20]:
layer.trainable = False


model.compile(optimizer=optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_flow, validation_data=val_flow, epochs=5, callbacks=[checkpoint, early])


print('Training finished. Model saved to', os.path.join(MODEL_DIR, 'model.h5'))