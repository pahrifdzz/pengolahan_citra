import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menampilkan gambar secara berdampingan
def show_images(images, titles, cmap=None):
    n = len(images)
    plt.figure(figsize=(20, 15))
    for i in range(n):
        plt.subplot(2, (n + 1) // 2, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) if len(images[i].shape) == 3 else images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Load gambar (warna asli)
image = cv2.imread("Ferrari.jpeg")

# Menu untuk memilih filter
print("Pilih filter yang ingin diterapkan:")
print("1. Peningkatan Kontras (Histogram Equalization)")
print("2. Pengurangan Noise (Gaussian Blur)")
print("3. Peningkatan Ketajaman (Sharpening)")
print("4. CLAHE (Peningkatan Kontras Lokal)")
print("5. Deteksi Tepi (Canny)")
print("6. Threshold Biner")
print("7. Adaptive Thresholding")
print("8. Dilasi")
print("9. Erosi")
print("10. Tampilkan Semua Filter")
choice = input("Masukkan nomor filter (pisahkan dengan koma untuk beberapa pilihan, misal: 1,2,3 atau masukkan 10 untuk semua): ")

# Proses filter
filters = choice.split(',')
processed_images = [image]
titles = ["Gambar Asli (Warna)"]

def apply_filters(image, option):
    """Menerapkan filter sesuai pilihan."""
    if option == "1":
        channels = cv2.split(image)
        equalized_channels = [cv2.equalizeHist(channel) for channel in channels]
        return cv2.merge(equalized_channels), "Peningkatan Kontras"
    elif option == "2":
        return cv2.GaussianBlur(image, (5, 5), 0), "Pengurangan Noise"
    elif option == "3":
        sharpening_kernel = np.array([[-1, -1, -1], 
                                      [-1,  9, -1], 
                                      [-1, -1, -1]])
        return cv2.filter2D(image, -1, sharpening_kernel), "Peningkatan Ketajaman"
    elif option == "4":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe_applied = clahe.apply(gray_channel)
        return clahe_applied, "CLAHE"
    elif option == "5":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray_image, 100, 200), "Deteksi Tepi (Canny)"
    elif option == "6":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return binary_thresh, "Threshold Biner"
    elif option == "7":
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return adaptive_thresh, "Adaptive Thresholding"
    elif option == "8":
        kernel = np.ones((5, 5), np.uint8)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return cv2.dilate(binary_thresh, kernel, iterations=1), "Dilasi"
    elif option == "9":
        kernel = np.ones((5, 5), np.uint8)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        return cv2.erode(binary_thresh, kernel, iterations=1), "Erosi"
    return None, None

# Jika semua filter dipilih
if "10" in filters:
    filters = [str(i) for i in range(1, 10)]  # Pilih semua filter

# Terapkan semua filter yang dipilih
for f in filters:
    f = f.strip()
    result, title = apply_filters(image, f)
    if result is not None:
        processed_images.append(result)
        titles.append(title)
    else:
        print(f"Pilihan tidak valid: {f}")

# Tampilkan hasil
show_images(processed_images, titles, cmap="gray")
1