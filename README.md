# DeepDent
**YOLOv8 ve Streamlit Tabanlı Diş Çürüğü ve Anomali Tespit Sistemi**

## Proje Hakkında
Bu çalışma, diş hekimliği alanında tanı süreçlerini yapay zeka teknolojileri ile desteklemek amacıyla geliştirilmiştir. Proje kapsamında, panoramik diş röntgenleri üzerinde derin öğrenme algoritmaları kullanılarak çürük (caries) ve çeşitli diş anomalilerinin otomatik tespiti hedeflenmiştir.

Geliştirilen sistem, yüklenen röntgen görüntülerini analiz ederek tespit edilen patolojik bölgeleri işaretler ve hekim için görsel bir ön değerlendirme sunar.

## Teknik Altyapı
Projenin geliştirilmesinde aşağıdaki teknolojiler ve kütüphaneler kullanılmıştır:

* **Python:** Temel programlama dili.
* **YOLOv8 (Ultralytics):** Nesne tespiti ve görüntü işleme için kullanılan derin öğrenme modeli.
* **Streamlit:** Modelin son kullanıcı tarafından kullanılabilmesi için geliştirilen web tabanlı arayüz.

## Çalışma Prensibi
1. Kullanıcı, sistem arayüzü üzerinden panoramik diş röntgeni görüntüsünü yükler.
2. Arka planda çalışan YOLOv8 modeli, görüntüyü işleyerek eğitildiği sınıflara (çürük, anomali vb.) ait öznitelikleri arar.
3. Tespit edilen bölgeler, güven skorları ile birlikte görüntü üzerinde işaretlenerek kullanıcıya sunulur.

---
**Geliştirici:** Beril Ay Kaplan
