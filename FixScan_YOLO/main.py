import cv2
from ultralytics import YOLO

# Eğitilmiş YOLOv8 modelinizi yükleyin (model yolunu kendi modelinize göre ayarlayın)
model = YOLO('last.pt')

# Sınıf isimlerini tanımlayın (modelinizin sınıf etiketlerine göre)
class_names = ["bad", "good"]  # Sınıf 0 -> "bad", Sınıf 1 -> "good"

# Görüntüyü yükle
image_path = './test_images/dnm.jpg'  # Test edeceğiniz görüntü dosyası
img = cv2.imread(image_path)

# Tahmin yap
results = model(img)

# Eşik değeri belirle
confidence_threshold = 0.5  # Güven eşiği (örnek: %50)

# Tahmin edilen kutular ve güven seviyeleri
boxes = results[0].boxes.xyxy.cpu().numpy()  # Koordinatlar (x1, y1, x2, y2)
confidences = results[0].boxes.conf.cpu().numpy()  # Güven seviyeleri
class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Sınıf ID'leri

# Yalnızca güven eşiğinin üzerindeki kutuları seç
filtered_boxes = []
for i, conf in enumerate(confidences):
    if conf >= confidence_threshold:
        filtered_boxes.append((boxes[i], conf, class_ids[i]))

# Görselleştirme
for box, conf, class_id in filtered_boxes:
    x1, y1, x2, y2 = map(int, box)  # Koordinatları tam sayıya çevir
    class_name = class_names[class_id]  # Sınıf adını alın
    label = f"{class_name} {conf:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Sarı kutu çiz
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Görüntüyü ekranda göster
cv2.imshow('Tahmin Sonuçları', img)

# Çıkış için bir tuşa basılmasını bekle
cv2.waitKey(0)
cv2.destroyAllWindows()
