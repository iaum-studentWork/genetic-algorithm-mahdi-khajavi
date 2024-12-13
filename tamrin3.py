import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. شبیه‌سازی داده‌ها
np.random.seed(42)
# ایجاد داده‌های تصادفی برای 200 مشتری
ages = np.random.randint(18, 70, size=200)  # سن مشتریان
incomes = np.random.randint(20000, 100000, size=200)  # درآمد مشتریان
purchases = np.random.randint(1, 50, size=200)  # تعداد خریدها
visit_times = np.random.randint(1, 60, size=200)  # زمان بازدید

# ساخت DataFrame
data = pd.DataFrame({
    'Age': ages,
    'Income': incomes,
    'Purchases': purchases,
    'Visit Time': visit_times
})

# 2. تعریف کلاس SOM
class SOM:
    def __init__(self, x_dim, y_dim, input_len, learning_rate=0.1, sigma=None):
        self.x_dim = x_dim  # ابعاد شبکه SOM (تعداد نودها در محور X)
        self.y_dim = y_dim  # ابعاد شبکه SOM (تعداد نودها در محور Y)
        self.input_len = input_len  # تعداد ویژگی‌ها
        self.learning_rate = learning_rate  # نرخ یادگیری
        self.sigma = sigma if sigma else max(x_dim, y_dim) / 2  # شعاع همسایگی
        # وزن‌های شبکه به صورت تصادفی ایجاد می‌شوند
        self.weights = np.random.random((x_dim, y_dim, input_len))  # وزن‌های شبکه

    def find_bmu(self, input_vector):
        # پیدا کردن بهترین واحد تطابق (BMU)
        bmu_idx = np.argmin(np.linalg.norm(self.weights - input_vector, axis=2))
        return np.unravel_index(bmu_idx, (self.x_dim, self.y_dim))

    def update_weights(self, input_vector, bmu_idx):
        # به‌روزرسانی وزن‌ها بر اساس ورودی و BMU
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                distance = np.linalg.norm(np.array([x, y]) - np.array(bmu_idx))
                influence = np.exp(-distance / (2 * (self.sigma ** 2)))  # تابع تاثیر
                # به‌روزرسانی وزن‌ها
                self.weights[x, y] += self.learning_rate * influence * (input_vector - self.weights[x, y])

    def train(self, data, num_iterations):
        # آموزش شبکه SOM
        for i in range(num_iterations):
            for input_vector in data:
                bmu_idx = self.find_bmu(input_vector)  # پیدا کردن BMU برای ورودی
                self.update_weights(input_vector, bmu_idx)  # به‌روزرسانی وزن‌ها

# 3. آماده‌سازی داده‌ها
features = data[['Age', 'Income', 'Purchases', 'Visit Time']].values
som = SOM(x_dim=10, y_dim=10, input_len=4)  # ایجاد یک SOM با ابعاد 10x10
som.train(features, num_iterations=100)  # آموزش شبکه

# 4. شناسایی خوشه‌ها
clusters = np.zeros((features.shape[0], 2))  # دو بعد برای ذخیره مختصات BMU

# برای هر مشتری، BMU را پیدا کرده و مختصات آن را ذخیره می‌کنیم
for i, input_vector in enumerate(features):
    bmu_idx = som.find_bmu(input_vector)
    clusters[i] = bmu_idx  # ذخیره مختصات BMU

# تبدیل خوشه‌ها به DataFrame برای راحتی تجزیه و تحلیل
clusters_df = pd.DataFrame(clusters, columns=['X', 'Y'])
data_with_clusters = pd.concat([data, clusters_df], axis=1)

# 5. نمایش خوشه‌ها با استفاده از نمودار
plt.figure(figsize=(12, 8))
scatter = plt.scatter(data_with_clusters['X'], data_with_clusters['Y'], c=data_with_clusters['Income'], cmap='viridis', s=100)
plt.colorbar(scatter, label='Income')
plt.title('Customer Clustering using SOM')
plt.xlabel('SOM X Coordinate')
plt.ylabel('SOM Y Coordinate')
plt.grid()

# 6. افزودن برچسب‌ها به نقاط
for i, row in data_with_clusters.iterrows():
    plt.annotate(f'ID: {i}', (row['X'], row['Y']), textcoords="offset points", xytext=(0, 5), ha='center')

plt.show()  # نمایش نمودار
