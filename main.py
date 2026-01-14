import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, input_size, hidden_size=4, output_size=1, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Инициализация весов
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        # Прямое распространение
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        # Binary Cross-Entropy Loss
        m = y_true.shape[0]
        return -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

    def backward(self, X, y):
        m = X.shape[0]

        # Обратное распространение ошибки
        dZ2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Обновление весов
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs=1000, verbose=True):
        losses = []
        for epoch in range(epochs):
            # Прямое распространение
            y_pred = self.forward(X)

            # Вычисление ошибки
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)

            # Обратное распространение
            self.backward(X, y)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses

    def predict(self, X, threshold=0.5):
        y_pred = self.forward(X)
        return (y_pred >= threshold).astype(int)

    def predict_proba(self, X):
        return self.forward(X)


class StudentAdmissionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def generate_sample_data(self, n_samples=200):
        """Генерация синтетических данных"""
        np.random.seed(42)

        # Средний балл (от 2 до 5)
        avg_grade = np.random.uniform(2, 5, n_samples)

        # Время подготовки (часы в неделю, от 0 до 30)
        study_hours = np.random.uniform(0, 30, n_samples)

        # Вероятность поступления (чем выше балл и время подготовки, тем выше вероятность)
        admission_prob = (avg_grade - 2) / 3 * 0.6 + (study_hours / 30) * 0.4
        admission = (admission_prob + np.random.normal(0, 0.1, n_samples)) > 0.5

        data = pd.DataFrame({
            'avg_grade': avg_grade,
            'study_hours': study_hours,
            'admitted': admission.astype(int)
        })

        return data

    def prepare_data(self, data):
        """Подготовка данных для обучения"""
        X = data[['avg_grade', 'study_hours']].values
        y = data['admitted'].values.reshape(-1, 1)

        # Нормализация данных
        X_scaled = self.scaler.fit_transform(X)

        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train, epochs=2000, learning_rate=0.1):
        """Обучение модели"""
        self.model = NeuralNetwork(
            input_size=X_train.shape[1],
            hidden_size=4,
            output_size=1,
            learning_rate=learning_rate
        )

        losses = self.model.train(X_train, y_train, epochs=epochs, verbose=True)
        return losses

    def evaluate(self, X_test, y_test):
        """Оценка модели"""
        if self.model is None:
            raise ValueError("Модель не обучена")

        y_pred = self.model.predict(X_test)
        accuracy = np.mean(y_pred == y_test)

        return accuracy, y_pred

    def predict_student(self, avg_grade, study_hours):
        """Предсказание для одного ученика"""
        if self.model is None:
            raise ValueError("Модель не обучена")

        # Подготовка входных данных
        student_data = np.array([[avg_grade, study_hours]])
        student_scaled = self.scaler.transform(student_data)

        # Получение предсказания
        proba = self.model.predict_proba(student_scaled)[0][0]
        prediction = self.model.predict(student_scaled)[0][0]

        return {
            'probability': proba,
            'admitted': bool(prediction),
            'avg_grade': avg_grade,
            'study_hours': study_hours
        }


# Пример использования
def main():
    # Создание и обучение модели
    predictor = StudentAdmissionPredictor()

    # Генерация данных
    print("Генерация данных...")
    data = predictor.generate_sample_data(300)

    # Подготовка данных
    X_train, X_test, y_train, y_test = predictor.prepare_data(data)

    print(f"Размер обучающей выборки: {X_train.shape[0]}")
    print(f"Размер тестовой выборки: {X_test.shape[0]}")

    # Обучение модели
    print("\nОбучение модели...")
    losses = predictor.train_model(X_train, y_train, epochs=2000, learning_rate=0.1)

    # Оценка модели
    accuracy, y_pred = predictor.evaluate(X_test, y_test)
    print(f"\nТочность на тестовой выборке: {accuracy:.2%}")

    # Предсказание для новых учеников
    print("\nПредсказания для тестовых учеников:")
    test_students = [
        (4.5, 25),  # Высокий балл, много занятий
        (3.0, 10),  # Средний балл, мало занятий
        (2.5, 5),  # Низкий балл, очень мало занятий
        (4.8, 28),  # Отличник, много занятий
    ]

    for grade, hours in test_students:
        result = predictor.predict_student(grade, hours)
        status = "поступит" if result['admitted'] else "не поступит"
        print(f"Средний балл: {grade:.1f}, Часы подготовки: {hours:.0f} -> "
              f"{status} (вероятность: {result['probability']:.1%})")

    # Визуализация процесса обучения
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Функция потерь во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Визуализация предсказаний
    plt.subplot(1, 2, 2)
    predictions = predictor.model.predict_proba(X_test)

    admitted_idx = y_test.flatten() == 1
    not_admitted_idx = y_test.flatten() == 0

    plt.scatter(predictions[admitted_idx], np.zeros_like(predictions[admitted_idx]),
                alpha=0.5, label='Поступил (реальность)')
    plt.scatter(predictions[not_admitted_idx], np.zeros_like(predictions[not_admitted_idx]),
                alpha=0.5, label='Не поступил (реальность)')

    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='Порог 0.5')
    plt.title('Распределение предсказанных вероятностей')
    plt.xlabel('Вероятность поступления')
    plt.yticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return predictor


main()