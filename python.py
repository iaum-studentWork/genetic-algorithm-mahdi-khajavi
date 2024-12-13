import numpy as np

# تعریف الگوهای س، ج، گ به صورت ماتریس 8x8
s = np.array([[0, 0, 1, 1, 1, 1, 0, 0],
              [0, 1, 0, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 0, 1, 0],
              [0, 0, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 1, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

j = np.array([[0, 0, 1, 1, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 0, 0],
              [0, 1, 0, 0, 1, 0, 0, 0],
              [0, 1, 0, 1, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

g = np.array([[0, 0, 1, 1, 1, 1, 0, 0],
              [0, 1, 0, 0, 0, 0, 1, 0],
              [0, 1, 0, 0, 0, 1, 1, 0],
              [0, 1, 0, 1, 0, 0, 0, 0],
              [0, 1, 1, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 1, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

# ترکیب داده‌ها و برچسب‌ها
data = np.array([s.flatten(), j.flatten(), g.flatten()])
labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # س، ج، گ

# تابع فعال‌سازی
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# مشتق تابع فعال‌سازی
def sigmoid_derivative(x):
    return x * (1 - x)

# پارامترها
input_size = 64  # 8x8
output_size = 3  # تعداد کلاس‌ها
learning_rate = 0.7
epochs = 10000

# وزن‌ها را به طور تصادفی مقداردهی اولیه می‌کنیم
np.random.seed(0)
weights = np.random.rand(input_size, output_size)

# آموزش شبکه
for epoch in range(epochs):
    # پیش‌بینی
    inputs = data
    predicted = sigmoid(np.dot(inputs, weights))
    
    # خطا
    error = labels - predicted
    
    # اصلاح وزن‌ها
    adjustments = learning_rate * np.dot(inputs.T, error * sigmoid_derivative(predicted))
    weights += adjustments

print("آموزش شبکه کامل شد.")

# ایجاد داده‌های گم شده
def introduce_noise(data, noise_level=0.2):
    noisy_data = data.copy()
    num_elements = noisy_data.size
    num_noisy_elements = int(num_elements * noise_level)
    indices = np.random.choice(num_elements, num_noisy_elements, replace=False)
    noisy_data.flat[indices] = 1 - noisy_data.flat[indices]  # تغییر روشن به تاریک و برعکس
    return noisy_data

# تست شبکه با داده‌های جدید
for i in range(len(data)):
    test_input = introduce_noise(data[i]).reshape(1, -1)
    prediction = sigmoid(np.dot(test_input, weights))
    predicted_class = np.argmax(prediction)
    print(f"الگوی {i + 1} (حرف {['س', 'ج', 'گ'][predicted_class]}): {prediction}")
