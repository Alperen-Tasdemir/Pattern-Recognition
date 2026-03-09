import numpy as np

class PCA:
    def __init__(self, n_components):
        """
        Principal Component Analysis (Temel Bileşenler Analizi)
        
        Parametreler:
        n_components (int): Tutulacak temel bileşen sayısı
        """
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Modeli uydurma adımı. Beklenen veri boyutu: (Örnek Sayısı, Özellik Sayısı)
        """
        # 1. Aşama: Veriyi merkezleme (Mean centering)
        # Her bir özelliğin (sütunun) ortalamasını çıkarıyoruz.
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Aşama: Kovaryans Matrisinin Hesaplanması
        # Veri boyutu: (Satır: Örnek, Sütun: Özellik)
        # Sütun bazlı özelliklerin kendi arasındaki varyans/kovaryanslarını bulmak için devriğini (T) uyduruyoruz
        cov_matrix = np.cov(X_centered.T)

        # 3. Aşama: Kovaryans matrisinin özdeğerleri (eigenvalues) ve özvektörleri (eigenvectors)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 4. Aşama: Özdeğerleri büyükten küçüğe sıralama
        # Özvektörler matrisin sütunlarında geldiği için satırlara alarak seçimi kolaylaştırıyoruz
        eigenvectors = eigenvectors.T
        
        # Özdeğerlerin büyüklüğüne göre azalan sırada indisleri al
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        # 5. Aşama: En büyük özdeğere karşılık gelen en iyi 'n' özvektörü (bileşeni) seçme
        self.components = eigenvectors[0:self.n_components]

    def transform(self, X):
        """
        Veriyi bulduğumuz ana bileşen uzayına izdüşürerek dönüştürür.
        """
        # Önce veriyi modele uydurduğumuz referansa göre merkezliyoruz
        X_centered = X - self.mean
        # İzdüşüm işlemi: (Örnek Sayısı, Özellik) x (Bileşen Sayısı, Özellik)^T
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        """
        Hem uydurur hem de veriyi dönüştürür.
        """
        self.fit(X)
        return self.transform(X)

# Sınıfımızın doğrudan denenmesi için bir kullanım örneği
if __name__ == "__main__":
    np.random.seed(42)
    # Örnek Sentetik Veri: 10 örnek, 4 özellik
    X_sample = np.random.rand(10, 4) * 10 
    
    # 2 boyuta düşürmek isteyelim
    pca = PCA(n_components=2)
    X_projected = pca.fit_transform(X_sample)
    
    print("Orijinal Veri Boyutu      :", X_sample.shape)
    print("Orijinal Veri Örneği      :\n", X_sample[:2])
    print("-" * 40)
    print("Sıkıştırılmış PCA Boyutu  :", X_projected.shape)
    print("PCA ile İzdüşürülmüş Veri :\n", X_projected[:2])
    print("-" * 40)
    print("Hesaplanan Ana Bileşenler (Varyans Vektörleri):\n", pca.components)
