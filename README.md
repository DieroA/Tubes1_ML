# Tugas Besar 1 IF3270 Pembelajaran Mesin

## Feed Forward Neural Network (FFNN)

### Deskripsi

Repository ini berisi implementasi Feed Forward Neural Network (FFNN) menggunakan Python tanpa library deep learning seperti PyTorch. Implementasi ini mencakup forward propagation, backward propagation, fungsi aktivasi, fungsi loss, serta metode penyimpanan dan pemuatan model.

### Setup

1. **Instalasi dependensi**
   Pastikan Python sudah terinstal, lalu install dependensi yang dibutuhkan:

   ```bash
   pip install numpy scikit-learn networkx matplotlib
   ```

2. **Konfigurasi model di `main.py`**
   Sesuaikan parameter model sesuai kebutuhan:
   - `depth`: Jumlah hidden layer dalam model
   - `width`: Jumlah neuron di dalam setiap hidden layer
   - `activation_func`: Fungsi aktivasi untuk setiap layer (ReLU, Sigmoid, Linear, Tanh, Softmax)
   - `loss_func`: Fungsi loss untuk menghitung error (CCE, MSE, BCE)
   - `weight_init_method`: Metode inisialisasi bobot (zero, uniform, normal)

### Menjalankan Program

1. Buka terminal di direktori root repository.
2. Jalankan `main.py` dengan perintah berikut:
   ```bash
   python -u src/main.py
   ```

### Pembagian Tugas

| Nama                       | NIM      | Tugas                                                                                   |
| -------------------------- | -------- | --------------------------------------------------------------------------------------- |
| **Rici Trisna Putra**      | 13522026 | Backward Propagation, Fungsi Loss, Fungsi Aktivasi, Save & Load, `main.py`              |
| **Imam Hanif Mulyarahman** | 13522030 | Forward Propagation, Regularisasi, Laporan                                               |
| **Diero Arga Purnama**     | 13522056 | Fungsi Aktivasi, Fungsi Loss, Kerangka Model (Neuron, Layer, FFNN), README, Visualisasi |
