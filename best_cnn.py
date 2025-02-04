# 01 데이터 불러오기: ImageDataGenerator(train, test)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# CIFAR-10 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
 
# 레이블을 one-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 02 데이터 전처리: Data Augmentation 추가
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,       # 이미지 회전
    width_shift_range=0.2,   # 가로 이동
    height_shift_range=0.2,  # 세로 이동
    shear_range=0.15,        # 전단 변환
    zoom_range=0.2,          # 확대/축소
    horizontal_flip=True,    # 좌우 반전
    validation_split=0.1     # 검증 데이터 분리
)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# 데이터 제너레이터 생성
train_generator = train_datagen.flow(x_train, y_train, batch_size=128, subset='training')
validation_generator = train_datagen.flow(x_train, y_train, batch_size=128, subset='validation')
test_generator = test_datagen.flow(x_test, y_test, batch_size=128)

# 04 신경망 모델 설정: Dense, Dropout, BatchNormalization, Regularization 추가
model = Sequential([
    # Convolutional Layer 1
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.0005), input_shape=(32, 32, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # Convolutional Layer 2
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Convolutional Layer 3
    Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Fully Connected Layer with Global Average Pooling
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')  # CIFAR-10은 10개의 클래스
])

from tensorflow.keras.optimizers import Adam

initial_learning_rate = 0.0005
optimizer = Adam(learning_rate=initial_learning_rate)

# 05 compile 설정: Optimizer와 Learning Rate Scheduler 추가
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning Rate Scheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)
# 06 fit(학습 진행)
epochs = 50  # 300 이하로 설정
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[lr_scheduler]
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
