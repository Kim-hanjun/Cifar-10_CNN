# 01 데이터 불러오기
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# CIFAR-10 데이터 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 레이블을 one-hot 인코딩
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 02 데이터 전처리: /255 (단순 정규화, 데이터 증강 없음)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 03 신경망 모델 설정
model = Sequential([
    # 간단한 Convolutional Layer 1
    Conv2D(16, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),

    # Convolutional Layer 2
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # Fully Connected Layer
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # CIFAR-10은 10개의 클래스
])

# 05 compile 설정
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 06 fit(학습 진행)
epochs = 10  # 간단한 모델을 위해 에포크를 줄임
history = model.fit(
    x_train, y_train,
    epochs=epochs,
    validation_split=0.1,
    batch_size=256  # 큰 배치 크기로 간단히 설정
)

# 테스트 데이터 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
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
    plt.savefig('low_performance_loss.png')  # 그래프 저장
    plt.show()
    
    # Accuracy 그래프
    plt.figure()
    plt.plot(epochs_range, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('low_performance_accuracy.png')  # 그래프 저장
    plt.show()

# 그래프 호출
plot_history(history)
