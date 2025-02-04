# 01 데이터 불러오기: ImageDataGenerator(train, test)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import RMSprop, Adam,SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# CIFAR-10 데이터 로드

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
 
# 레이블을 one-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 02 데이터 전처리: Data Augmentation 추가
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.1,
)


test_datagen = ImageDataGenerator(rescale=1.0/255)

# 데이터 제너레이터 생성
train_generator = train_datagen.flow(x_train, y_train, batch_size=256, subset='training')
validation_generator = train_datagen.flow(x_train, y_train, batch_size=256, subset='validation')
test_generator = test_datagen.flow(x_test, y_test, batch_size=256)

# 04 신경망 모델 설정: Dense, Dropout, BatchNormalization, Regularization 추가
weight_decay = 1e-4

model = Sequential()

model.add(Conv2D(32, 
                 (3, 3), 
                 padding='same',
                 kernel_regularizer=l2(weight_decay),
                 input_shape=(32, 32, 3),
                 activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(32,
                 (3, 3),
                 padding='same',
                 kernel_regularizer=l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64,
                 (3, 3),
                 padding='same',
                 kernel_regularizer=l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, 
                 (3, 3),
                 padding='same',
                 kernel_regularizer=l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, 
                 (3, 3),
                 padding='same',
                 kernel_regularizer=l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, 
                 (3, 3),
                 padding='same',
                 kernel_regularizer=l2(weight_decay),
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(len(CIFAR10_CLASSES), activation='softmax')) # 분류할 FC Layer 추가


# 05 compile 설정: Optimizer와 Learning Rate Scheduler 추가
model.compile(loss=categorical_crossentropy, 
              optimizer=RMSprop(learning_rate=0.0001), 
              metrics=['accuracy'])
def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate

# 06 fit(학습 진행)
epochs = 50  # 300 이하로 설정
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    validation_split=0.1,
    batch_size=256  # 큰 배치 크기로 간단히 설정
)

# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# 학습 및 검증 손실, 정확도 그래프 그리기
def plot_history(history):
    epochs_range = range(len(history.history['loss']))
    
    # Loss 그래프
    plt.figure()
    plt.plot(epochs_range, history.history['loss'], label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_validation_loss.png')  # 파일 저장
    plt.show()
    
    # Accuracy 그래프
    plt.figure()
    plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('training_validation_accuracy.png')  # 파일 저장
    plt.show()

# 그래프 호출
plot_history(history)
