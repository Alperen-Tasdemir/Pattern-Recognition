import numpy as np

class ICA:
    def __init__(self, n_components=None, max_iter=200, tol=1e-4):
        """
        Independent Component Analysis (Bağımsız Bileşenler Analizi - FastICA algoritması)
        
        Parametreler:
        n_components (int): Ayrıştırılacak kaynak (source) sayısı
        max_iter (int): En fazla yapılacak iterasyon sayısı (yakınsama için)
        tol (float): Yakınsama tolerans noktası
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.W = None  # Karışımı çözecek ayrıştırma (unmixing) matrisi

    def _center(self, X):
        """Veriyi özelliklerin ortalaması 0 olacak şekilde yatay eksende merkezler."""
        mean = np.mean(X, axis=1, keepdims=True)
        return X - mean, mean

    def _whiten(self, X):
        """
        Veriyi beyazlaştırma işlemi (Whitening). 
        Amaç değişkenlerin varyansını 1 yapmak ve birbirleriyle korale olmamalarını sağlamaktır.
        """
        cov = np.cov(X)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Çok küçük özdeğerlerin hesaplamayı bozmasını önlemek adına filtreliyoruz
        eps = 1e-7
        idx = eigenvalues > eps
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
        
        # PCA Temelli Beyazlaştırma Matrisi (Whitening Matrix)
        white_M = np.dot(D_inv_sqrt, eigenvectors.T)
        X_whitened = np.dot(white_M, X)
        
        return X_whitened, white_M

    def _g(self, x):
        """Aktivasyon/Negentropi fonksiyonu (Non-linear). Genellikle tanh tercih edilir."""
        return np.tanh(x)

    def _g_prime(self, x):
        """Aktivasyon fonksiyonunun (tanh) birinci mertebeden türevi."""
        return 1.0 - np.tanh(x)**2

    def fit_transform(self, X):
        """
        FastICA modelini uydurur ve kaynak bileşenlerini hesaplayıp geri döner.
        X'in şekli standart olarak (Özellik Sayısı, Örnek Sayısı) şeklinde olmalıdır. Model içeride böyle işler.
        """
        # Veriyi satıra göre transpoze ediyoruz çünkü FastICA (Özellik/Sensör x Zaman/Örnek) şeklinde çalışır
        X = X.T 

        n_features, n_samples = X.shape
        if self.n_components is None:
            self.n_components = n_features

        # 1. Aşama: Veriyi Merkezle
        X_centered, self.mean_ = self._center(X)
        
        # 2. Aşama: Beyazlaştırma İşlemi (X' ve V matrisi bulma)
        X_whitened, self.whiten_M_ = self._whiten(X_centered)
        n_whitened_features = X_whitened.shape[0]

        # 3. Aşama: ICA Ağırlıklarını Rastgele Başlat
        W = np.random.rand(self.n_components, n_whitened_features)
        
        # FastICA Algoritması
        for i in range(self.n_components):
            w = W[i, :]
            w = w / np.linalg.norm(w) # Normalize et
            
            for _ in range(self.max_iter):
                # Yaklaşım (Update kuralı)
                # w = E[xg(w^T x)] - E[g'(w^T x)] * w
                w_tx = np.dot(w, X_whitened)
                term1 = (X_whitened * self._g(w_tx)).mean(axis=1)
                term2 = self._g_prime(w_tx).mean() * w
                w_new = term1 - term2
                        
                # Birbirine karışmayı (korelasyonu) önleme - Dekorelasyon(Gram-Schmidt methodu)
                if i > 0:
                    w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
                
                # Tekrar normalize et
                w_new = w_new / np.linalg.norm(w_new)
                
                # Yakınsama(Convergence) durumu kontrolü
                # 1'e veya -1'e yaklaşması paralelliği ifade eder.
                distance = np.abs(np.abs((w * w_new).sum()) - 1)
                w = w_new
                
                if distance < self.tol:
                    break
                    
            # Güncellenmiş seriyi asıl ağırlık matrisine ekle
            W[i, :] = w
        
        self.W = W
        
        # Orijinal sinyalleri(kaynakları) çözümleyerek bul (S = W * X_whitened)
        S = np.dot(self.W, X_whitened)
        
        # Tekrardan X standart formatına yani (Örnek Sayısı, Özellik Sayısı) boyutlarına döndürüyoruz
        return S.T 

# Sınıfımızın doğrudan denenmesi için bir kullanım örneği
if __name__ == "__main__":
    # Örnek Sentetik Karışık Veri Oluşturma (Cocktail Party Problem örneği)
    np.random.seed(42)
    zaman = np.linspace(0, 8, 2000)
    
    # 1. Orijinal Kaynak: Sinüs dalgası
    s1 = np.sin(2 * zaman) 
    # 2. Orijinal Kaynak: Kare dalga
    s2 = np.sign(np.sin(3 * zaman)) 
    
    S_orijinal = np.c_[s1, s2] # İki sinyali birleştir (2000, 2)
    
    # Ortama 2 adet mikrofon koyduğumuzu varsayalım. Sinyaller ortama karışıp yayılıyor
    karistirma_matrisi = np.array([[1, 1], [0.5, 2]])
    X_karisik = np.dot(S_orijinal, karistirma_matrisi.T) # Karıştırılmış Sinyal (2000, 2)
    
    # ICA ile bu sinyalleri saf hallerine birbirinden ayrı biçimde ayırmaya çalışıyoruz
    ica = ICA(n_components=2)
    S_ayrilmis = ica.fit_transform(X_karisik)
    
    print("ICA Çözümlemesi Tamamlandı.")
    print("Orijinal Kaynak Boyutu      :", S_orijinal.shape)
    print("Karışık Sinyal (Gözlem)     :", X_karisik.shape)
    print("Ulaşılan (Ayrışmış) Sinyal  :", S_ayrilmis.shape)
    
    # Not: ICA, sinyallerin sırasını (index) veya ölçeğini (scale) korumaz,
    # Sadece sinyal formlarını birbirinden bağımsızlaştırır.
