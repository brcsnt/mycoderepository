import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Gerekli kütüphanelerin içe aktarılması
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# 1. Veri Kümesini Yükleme ve Ön İşleme
# Veri kümesinin bulunduğu dizin
dataset_directory = '/path/to/dataset'

# Resimleri ve etiketleri saklamak için listeler
images = []
ages = []
genders = []

# Dosya isimlerini okuyarak veri kümesini yükleme
for filename in os.listdir(dataset_directory):
    if filename.endswith(".jpg"):  # veya uygun dosya uzantısı
        image = load_img(os.path.join(dataset_directory, filename), target_size=(200, 200))
        image = img_to_array(image)
        images.append(image)

        # Dosya isminden yaş, cinsiyet ve diğer bilgileri ayıklama
        parts = filename.split('_')
        age = int(parts[0])
        gender = int(parts[1])  # 0 ve 1 değerleri alabilir

        ages.append(age)
        genders.append(gender)

# Görüntü verilerini numpy dizisine dönüştürme ve normalize etme
images = np.array(images, dtype="float32") / 255.0
ages = np.array(ages)
genders = np.array(genders)

from sklearn.model_selection import train_test_split

# Veri setini %70 eğitim ve %30 test olarak ayırma
x_train, x_test, y_train_age, y_test_age = train_test_split(images, ages, test_size=0.3, random_state=42)
_, _, y_train_gender, y_test_gender = train_test_split(images, genders, test_size=0.3, random_state=42)


# 2. Kendi CNN Mimarinizi Oluşturma
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def create_custom_cnn():
    input = Input(shape=(200, 200, 3))
    
    # Convolutional layers
    x = Conv2D(32, (3, 3), activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Multi-task outputs
    age_output = Dense(1, name='age_output')(x)  # Yaş tahmini
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)  # Cinsiyet tahmini

    model = Model(inputs=input, outputs=[age_output, gender_output])
    return model
from tensorflow.keras.applications import VGG16

# 3. Transfer Learning Modeli
def create_transfer_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    base_model.trainable = False  # Temel modelin katmanlarını dondur

    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)

    age_output = Dense(1, name='age_output')(x)
    gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)

    model = Model(inputs=base_model.input, outputs=[age_output, gender_output])
    return model

from tensorflow.keras.optimizers import Adam

# Özel CNN modelini derleme ve eğitme
custom_cnn = create_custom_cnn()
custom_cnn.compile(optimizer=Adam(),
                   loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
                   metrics={'age_output': 'mae', 'gender_output': 'accuracy'})
history_custom_cnn = custom_cnn.fit(x_train, {'age_output': y_train_age, 'gender_output': y_train_gender},
                                    epochs=10, batch_size=32, validation_split=0.2)

# Transfer öğrenme modelini derleme ve eğitme
transfer_model = create_transfer_model()
transfer_model.compile(optimizer=Adam(),
                       loss={'age_output': 'mse', 'gender_output': 'binary_crossentropy'},
                       metrics={'age_output': 'mae', 'gender_output': 'accuracy'})
history_transfer_model = transfer_model.fit(x_train, {'age_output': y_train_age, 'gender_output': y_train_gender},
                                            epochs=10, batch_size=32, validation_split=0.2)


# Model Performansının Değerlendirilmesi
import matplotlib.pyplot as plt

def plot_model_performance(history, title):
    plt.figure(figsize=(12, 5))

    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['gender_output_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_gender_output_accuracy'], label='Validation Accuracy')
    plt.title(title + ' - Gender Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['age_output_loss'], label='Training Loss')
    plt.plot(history.history['val_age_output_loss'], label='Validation Loss')
    plt.title(title + ' - Age Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Her iki modelin performansını çizdir
plot_model_performance(history_custom_cnn, 'Custom CNN')
plot_model_performance(history_transfer_model, 'Transfer Learning Model')
