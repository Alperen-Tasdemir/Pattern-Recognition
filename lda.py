import numpy as np

class LDA:
    def __init__(self, n_components):
        """
        Linear Discriminant Analysis (Doğrusal Diskriminant Analizi)
        
        Parametreler:
        n_components (int): Tutulacak bileşen sayısı (genellikle sınıf_sayısı - 1 kadardır)
        """
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        """
        Modeli eğitiyoruz. LDA gözetimli(supervised) bir algoritmadır, 
        dolayısıyla X'in yanında sınıflara (y) de ihtiyacımız var.
        """
        n_features = X.shape[1]
        class_labels = np.unique(y)

        # Sw: Sınıf İçi Dağılım Matrisi (Within-class scatter matrix)
        # Sb: Sınıflar Arası Dağılım Matrisi (Between-class scatter matrix)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        # Tüm verinin ortalaması
        mean_overall = np.mean(X, axis=0)

        for c in class_labels:
            # Sadece bu sınıfa (c) ait verileri al
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)

            # --- Sınıf içi scatter matrisini oluşturma ---
            # (X_c - mean_c)'nin karesel çarpımlarının toplamı
            X_c_centered = X_c - mean_c
            S_W += np.dot(X_c_centered.T, X_c_centered)

            # --- Sınıflar arası scatter matrisini oluşturma ---
            n_c = X_c.shape[0]
            # Sınıf ortalaması ile genel ortalama arasındaki fark
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff.dot(mean_diff.T))

        # LDA amacı: Sınıf içi değişimi minimum, sınıflar arası değişimi maksimum yapmaktır.
        # Maximize edilecek oran: Sw^-1 * Sb
        A = np.linalg.inv(S_W).dot(S_B)
        
        # Bu matrisin özdeğer ve özvektörlerini hesaplıyoruz
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # Özvektörleri kolay ulaşmak için transpoze ediyoruz (satırlara taşıyoruz)
        eigenvectors = eigenvectors.T
        
        # Özdeğerlerin mutlak değerlerine göre büyükten küçüğe indislerini sıralıyoruz
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # En iyi olan diskriminantları (ayrıştırıcı özellikleri) bileşen sayımız kadar saklıyoruz
        self.linear_discriminants = eigenvectors[0:self.n_components]

    def transform(self, X):
        """
        Gelen veri setini eğitilmiş ayırıcı diskriminantlara göre izdüşürür.
        """
        return np.dot(X, self.linear_discriminants.T)
        
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

# Sınıfımızın doğrudan denenmesi için bir kullanım örneği
if __name__ == "__main__":
    # Örnek Sentetik Veri
    # X: (8, 2) boyutunda 8 örnek. 
    # İlk 4 tanesi Sınıf 0'a, diğer 4 tanesi Sınıf 1'e ait.
    X_sample = np.array([
        [1.5, 2.0], [2.0, 2.5], [1.0, 1.0], [2.5, 3.0], 
        [8.0, 8.5], [9.5, 9.0], [8.5, 10.0],[10.0, 9.5]
    ])
    y_sample = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    # 1 boyuta (yeni bir eksene) projeksiyon yapacağız
    lda = LDA(n_components=1)
    X_projected = lda.fit_transform(X_sample, y_sample)

    print("Orijinal Veriler (X):\n", X_sample)
    print("Orijinal Sınıflar (y):\n", y_sample)
    print("-" * 40)
    print("Sınıfları Ayrıştıran En İyi Eksene (1D) İzdüşümü:")
    print(X_projected)
