# TEKNOFEST 2026 - HAVACILIKTA YAPAY ZEKA YARIŞMASI
## Tüm Dokümantasyon İndeksi

---

## 📋 Dosya Listesi

Bu klasörde TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması ile ilgili tüm resmi dokümantasyon Markdown formatında düzenlenmiş olarak bulunmaktadır.

### 1. **01_HAVACILIKTA_YAPAY_ZEKA_GENEL_SARTNAME.md**
   - **İçerik:** Yarışmanın genel kuralları, amaçları, katılım koşulları
   - **Bölümler:**
     - Yarışmanın Amacı
     - Yarışmanın Kapsamı (3 Görev)
     - Yarışma Başvurusu
     - Yarışma Detayları
     - Puanlandırma
     - Yarışma Takvimi
     - Ödüller
     - Katılım Koşulları
     - İletişim Bilgileri
   - **Hedef Kitle:** Tüm yarışmacılar

### 2. **02_HAVACILIKTA_YAPAY_ZEKA_TEKNIK_SARTNAME.md**
   - **İçerik:** Yarışmanın teknik detayları, algoritma gereksinimleri, puanlama kriterleri
   - **Bölümler:**
     - Görevlerin Detaylı Açıklaması
       - Birinci Görev: Nesne Tespiti
       - İkinci Görev: Pozisyon Tespiti
       - Üçüncü Görev: Görüntü Eşleme
     - Yarışma Oturumları
     - Teknik Sunum
     - Raporlama
     - Çevrimiçi Yarışma Simülasyonu
     - Yazılım ve Donanım Özellikleri
     - Sunucu Bağlantı Protokolü
     - JSON Formatları
     - Puanlama Kriterleri ve Formüller
   - **Hedef Kitle:** Teknik ekip, yazılım geliştirici

### 3. **03_ON_TASARIM_RAPORU_SABLONU.md**
   - **İçelik:** Ön Tasarım Raporu yazım şablonu ve yönergeleri
   - **Bölümler:**
     - Takım Şeması
     - Proje Mevcut Durum Değerlendirmesi
     - Algoritmalar ve Sistem Mimarisi
     - Özgünlük
     - Proje Takvimi
     - Sonuçlar ve İnceleme
     - Kaynakça
     - Genel Rapor Düzeni
   - **Puan Dağılımı:** 100 puan (10+30+10+10+10+30+5+5)
   - **Hedef Kitle:** Rapor yazacak takımlar

---

## 🎯 Yarışma Görevleri Özeti

### Birinci Görev: Nesne Tespiti (Puan: %25)
- **Amaç:** Hava aracı kamerası görüntülerinden nesneleri tespit etmek
- **Tespit Edilecek Nesneler:**
  - Taşıt (Hareket Durumu: Hareketli/Hareketsiz)
  - İnsan
  - Uçan Araba Park (UAP) Alanı (İniş Durumu: Uygun/Uygun Değil)
  - Uçan Ambulans İniş (UAİ) Alanı (İniş Durumu: Uygun/Uygun Değil)
- **Puanlama:** mAP (mean Average Precision) metriği, IoU eşik değeri: 0.5

### İkinci Görev: Pozisyon Tespiti (Puan: %40)
- **Amaç:** GPS sistemi devre dışı olduğunda görüntü verilerinden pozisyon kestirimi
- **Çıktı:** X, Y, Z eksenlerindeki metre cinsinden yer değiştirme
- **Puanlama:** Ortalama hata hesaplaması

### Üçüncü Görev: Görüntü Eşleme (Puan: %25)
- **Amaç:** Daha önce tanımlanmamış nesneleri tespit etme
- **Özellik:** Farklı kameralar, açılar, irtifalar, görüntü işlemelerden geçmiş nesneler
- **Puanlama:** mAP metriği

---

## 📅 Önemli Tarihler

| Tarih | Etkinlik |
|-------|----------|
| 16.02.2026 | Teknik Şartnamenin İlanı |
| 28.02.2026 | Yarışma Son Başvuru Tarihi |
| 10-28.03.2026 | Örnek Eğitim Videosunun Teslimi |
| 22.04.2026 | Ön Tasarım Raporu Son Teslim Tarihi |
| 22.05.2026 | 1. Ön Eleme Sonuçları |
| 01-06.06.2026 | Takımlarla Soru-Cevap Toplantısı |
| 09.07.2026 | Çevrim İçi Yarışma Simülasyonu |
| 17.07.2026 | 2. Ön Eleme Sonuçları |
| Ağustos-Eylül 2026 | Yarışma Finalleri |
| 30 Eylül-4 Ekim 2026 | TEKNOFEST |

---

## 💰 Ödüller

### Para Ödülleri:
- **Birincilik:** 250.000 ₺ (Danışman: 15.000 ₺)
- **İkincilik:** 225.000 ₺ (Danışman: 12.000 ₺)
- **Üçüncülük:** 200.000 ₺ (Danışman: 10.000 ₺)

### Prestij Ödülleri:
- **En İyi Sunum Ödülü**
- **Yenilikçi Yazılım Ödülü**

---

## 👥 Takım Gereksinimleri

- **Takım Büyüklüğü:** Minimum 2, Maksimum 5 kişi
- **Danışman:** Lise takımları için zorunlu, üniversite takımları için isteğe bağlı
- **İletişim Sorumlusu:** Her takımda 1 kişi
- **Takım Kaptanı:** Zorunlu

---

## 🔧 Teknik Gereksinimler

### Donanım:
- Ethernet bağlantı girişi zorunlu
- Saniyede 1 görüntü karesi işleyebilecek kapasite yeterli
- Algoritma hızı puanlandırma kriteri değildir

### Yazılım:
- İstenilen işletim sistemi kullanılabilir
- İstenilen programlama dili kullanılabilir
- Çevrimiçi hizmetler yasaktır (internet bağlantısı yasak)

### Veri Formatı:
- Video: 5 dakika, 7.5 FPS, 2250 görüntü karesi
- Çözünürlük: Full HD veya 4K
- Haberleşme: JSON formatında API

---

## 📊 Puanlama Dağılımı

| Bileşen | Puan Oranı |
|---------|------------|
| Birinci Görev (Nesne Tespiti) | %25 |
| İkinci Görev (Pozisyon Tespiti) | %40 |
| Üçüncü Görev (Görüntü Eşleme) | %25 |
| Final Tasarım Raporu | %5 |
| Yarışma Sunumu | %5 |
| **Toplam** | **%100** |

**Başarı Kriteri:** Tüm oturumlarda elde edilen puan yüzdelerinin ortalaması %70'i geçmesi

---

## 📞 İletişim

- **Genel Sorular:** TEKNOFEST web sitesi - Havacılıkta Yapay Zeka Yarışması grubu
- **Organizasyonel Sorular:** iletisim@teknofest.org
- **Github:** Kod blokları ve örnek veri setleri
- **Google Groups:** Takımlar arası iletişim ve destek

---

## 📝 Notlar

- Tüm dokümantasyon Türkçe dilinde hazırlanmıştır
- Markdown formatı, kolay okunabilirlik ve düzenleme için seçilmiştir
- Orijinal PDF ve DOCX dosyaları korunmuştur
- Tüm bilgiler TEKNOFEST 2026 resmi kaynaklarından alınmıştır

---

**Son Güncelleme:** 20 Nisan 2026

*Bu dokümantasyon TEKNOFEST 2026 Havacılıkta Yapay Zeka Yarışması için hazırlanmıştır.*
