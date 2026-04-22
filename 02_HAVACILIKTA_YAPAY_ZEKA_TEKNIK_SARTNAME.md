# HAVACILIKTA YAPAY ZEKA YARIŞMASI - TEKNİK ŞARTNAME

## VERSİYONLAR

| VERSİYON | TARİH | Açıklama |
|----------|-------|---------|
| V1.0 | 21.02.2026 | TEKNOFEST 2026 İlk Versiyon |
| V1.1 | 20.04.2026 | 5.1. Güncellemesi |

---

## TANIMLAR VE KISALTMALAR DİZİNİ

- **KYS:** TEKNOFEST Kurumsal Yönetim Sistemi
- **Takım Kaptanı:** Takımın organizasyonundan sorumlu olan ve süreçlerde liderlik görevini üstlenen kişi
- **Takım Danışmanı:** Her takım için en fazla bir (1) öğretmen/eğitmen/akademisyen
- **TEKNOFEST:** Havacılık, Uzay ve Teknoloji Festivali
- **T3 Vakfı:** Türkiye Teknoloji Takımı Vakfı
- **Yarışma Süreci:** Yarışma başvurularının alınmaya başladığı tarih ile final sonuçlarının açıklandığı tarih arasında geçen süre

---

## İÇİNDEKİLER

1. [GİRİŞ](#1-giriş)
2. [GÖREVLER](#2-görevler)
   - 2.1. Birinci Görev: Nesne Tespiti
   - 2.2. İkinci Görev: Pozisyon Tespiti
   - 2.3. Üçüncü Görev: Görüntü Eşleme
3. [YARIŞMA](#3-yarisma)
4. [TEKNİK SUNUM](#4-teknik-sunum)
5. [RAPORLAMA](#5-raporlama)
6. [ÇEVRİMİÇİ YARIŞMA SİMÜLASYONU](#6-cevrimici-yarisma-simulasyonu)
7. [TAKIMLARIN YAZILIM VE DONANIM ÖZELLİKLERİ](#7-takimlarin-yazilim-ve-donanim-ozellikleri)
8. [YARIŞMA SIRASINDA SUNUCU İLE BAĞLANTI](#8-yarisma-sirasinda-sunucu-ile-baglanti)
9. [PUANLAMA](#9-puanlama)
10. [YARIŞMA GİTHUB ve GOOGLE GROUPS SAYFALARI](#10-yarisma-github-ve-google-groups-sayfalari)
11. [YARIŞMA SONUÇLARININ DUYURULMASI VE ÖDÜLLENDİRME](#11-yarisma-sonuclarinin-duyurulmasi-ve-odullendirme)

---

## 1. GİRİŞ

Bu doküman Havacılıkta Yapay Zekâ Yarışması öncesi ve yarışma sırasında yarışmacıların bilgisi dahilinde olması gereken durumları içermektedir.

---

## 2. GÖREVLER

TEKNOFEST 2026 Havacılıkta Yapay Zekâ yarışması kapsamında yarışmacılar **üç farklı görevi** yerine getirecek algoritmalar geliştirmelidir:

1. **Nesne Tespiti**
2. **Pozisyon Kestirimi**
3. **Görüntü Eşleme**

Yarışmacılar, hava aracının alt-görüş kamerasından aldıkları görüntüleri kendi geliştirdikleri algoritmalar ile işleyerek:
- İlk görev için görüntü karesindeki belirli nesneleri ve bu nesnelere ait hareket durumlarını tespit etmeli
- İkinci görev için hava aracının zamana bağlı olarak pozisyon bilgisini kestirmeli
- Üçüncü görev için görev başlangıcında paylaşılan referans nesneleri hava aracı görüntülerinden tespit etmelidir

### 2.1. Birinci Görev: Nesne Tespiti

Havacılıkta Yapay Zekâ yarışması kapsamında yarışmacılar tarafından tespit edilmesi beklenen nesne türleri:

1. **Taşıt**
2. **İnsan**
3. **Uçan Araba Park (UAP)** alanı
4. **Uçan Ambulans İniş (UAİ)** alanı

#### Teknik Bilgiler:

- Videolar hava aracının kalkışını, inişini ve seyrüseferini içerebilir
- Her oturumda 5 dakikalık video, 7.5 FPS ile toplam **2250 görüntü karesi** verilecektir
- Videolar Full HD veya 4K çözünürlüğünde çekilmektedir
- Video kareleri herhangi bir görüntü formatında olabilir (jpg, png vs.)
- Kamera açısı: 70-90 derece aralığında değişken olacaktır
- Hava aracı kar, yağmur vb. hava koşullarında uçabilir
- Hava aracı şehir, orman ve deniz üzerinde uçabilir
- Görüntülerde bozulmalar (bulanıklık, ölü pikseller) olabilir
- Görüntüler RGB veya termal kamera ile elde edilmiş olabilir

#### 2.1.1. Taşıt ve İnsan Tespiti

**Taşıt Sınıfı (ID: 0)**
- Hareket Durumu: 0 (Hareketsiz) veya 1 (Hareketli)
- İniş Durumu: -1 (Taşıt Değil)

Taşıt olarak değerlendirilen nesneler:
- Motorlu karayolu taşıtları (Otomobiller, Motosikletler, Otobüsler, Kamyonlar, Traktör, ATV vb.)
- Raylı taşıtlar (Trenler, Lokomotifler, Vagonlar, Tramvaylar, Monoraylar, Füniküler)
- Tüm deniz taşıtları

**İnsan Sınıfı (ID: 1)**
- Hareket Durumu: -1
- İniş Durumu: -1
- Ayakta duran ya da oturan fark etmeksizin tüm insanlar

**Önemli Kurallar:**
- Tren olması durumunda lokomotif ve vagonların her biri ayrı bir obje olarak tanımlanmalıdır
- Tamamı görünmeyen taşıt ve insan nesnelerinin de tespit edilmesi beklenmektedir
- Bisiklet ve motosiklet sürücüleri "insan" olarak etiketlenmemelidir
- Scooter, sürücüsü olmadığı zamanlarda taşıt, sürücüsü olduğu zamanlarda ise insan olarak etiketlenmelidir
- Kamera hareketinden kaynaklanan yalancı hareket durumları ayırt edilebilmelidir

#### 2.1.2. UAP ve UAI Tespiti

**Uçan Araba Park (UAP) Alanı (ID: 2)**
- Hareket Durumu: -1
- İniş Durumu: 0 (Uygun Değil), 1 (Uygun)
- 4,5 metre çapında bir daire ile belirtilmektedir

**Uçan Ambulans İniş (UAİ) Alanı (ID: 3)**
- Hareket Durumu: -1
- İniş Durumu: 0 (Uygun Değil), 1 (Uygun)
- 4,5 metre çapında bir daire ile belirtilmektedir

**İniş Durumu Kuralları:**
- Alanların üzerinde herhangi bir cisim bulunuyorsa iniş için uygun değildir
- İniş durumunun "uygun" olabilmesi için alanların tamamının kare içinde bulunması gerekmektedir

#### 2.1.3. Algoritma Çalışma Şartları

- Yarışmacılar sunucu ile bağlantı kurup istek gönderdiklerinde bir adet görüntü karesi alacaklardır
- Her görüntü karesinde tespit ettikleri nesnelerin bilgisini istenen formatta sunucuya yollayacaklardır
- Herhangi bir kare için sonuç göndermeden sıradaki karenin alınması için istek gönderilemez
- Her görüntü karesine 1 adet sonuç yollanmalıdır
- Belirlenen limit değerden fazla sonuç yollayan takımların sonuç gönderme kabiliyetleri engellenebilir

### 2.2. İkinci Görev: Pozisyon Tespiti

İkinci görevde hava aracının konumlandırma sisteminin kullanılamaz veya güvenilemez hale geldiği durumlar simüle edilecek ve sadece görüntü verileri üzerinden pozisyon kestirimi yapılması beklenecektir.

#### 2.2.1. Pozisyon Tespiti

- Yarışmacılar geliştirdikleri pozisyon kestirimi algoritmaları ile verilen kamera görüntülerini kullanarak hava aracının referans koordinat sistemindeki pozisyonunu kestireceklerdir
- Referans koordinat sisteminde ilk pozisyon bilgisi: x₀=0.00 [m], y₀=0.00 [m], z₀=0.00 [m]
- Kamera açısı: 70-90 derece aralığında
- Hava aracının kamera parametre bilgileri yarışmacılarla paylaşılacaktır

#### 2.2.2. Algoritma Çalışma Şartları

**Yarışmacıların Sunucu İsteği Sonrasında Alacağı Bilgiler:**

| Başlık | Detay |
|--------|-------|
| Video Karesi Bilgisi | Benzersiz isim |
| Pozisyon Bilgisi - X | X eksenindeki metre cinsinden yer değiştirme |
| Pozisyon Bilgisi - Y | Y eksenindeki metre cinsinden yer değiştirme |
| Pozisyon Bilgisi - Z | Z eksenindeki metre cinsinden yer değiştirme |
| Pozisyon Bilgisi - Sağlık | Pozisyon tespit sisteminin sağlıklı çalışıp çalışmadığı |

- Sağlık değeri 1 ise: Yarışmacı referans değeri değiştirmeden gönderebilir
- Sağlık değeri 0 ise: Yarışmacı kendi algoritması ile kestirdiği pozisyon bilgisini göndermelidir
- İlk 1 dakikada (450 kare) pozisyon bilgisi sağlıklı olarak alınacaktır
- Son 4 dakikada (1800 kare) pozisyon bilgisi sağlıksız duruma geçebilir

### 2.3. Üçüncü Görev: Görüntü Eşleme

Üçüncü görev, hava araçlarının daha önce tanımlanmamış nesneleri görsel veri üzerinden anlık olarak tanıma ve takip etme yeteneğini test etmektedir.

- Oturum başlangıcında belirli sayıda ve farklı zorluk seviyelerinde referans nesne görüntüsü paylaşılacaktır
- Yarışmacılar görüntü akışı esnasında verilen referanslardan tespit ettikleri nesnelerin koordinatlarını sonuçları ile beraber sunucuya göndereceklerdir
- Oturum başında verilen referans nesnelerin tamamı oturum içerisindeki görüntülerde mevcut olmayabilir

**Oturum esnasında paylaşılan görüntüler:**
- Farklı kameradan çekilmiş olabilir (örn. termal → RGB)
- Farklı bir açıdan veya irtifadan çekilmiş olabilir
- Uydu görüntüleri üzerinden alınmış bir nesne olabilir
- Yer yüzeyinden çekilmiş nesneler olabilir
- Çeşitli görüntü işleme işlemlerinden geçmiş olabilir

---

## 3. YARIŞMA

### Yarışma Kuralları:

- Ön Tasarım Raporunu teslim etmiş ve Çevrimiçi Yarışma Simülasyonundan yeterli puanı alan takımlar TEKNOFEST 2026'da yarışmak için hak kazanacaktır
- Yarışma alanında yerel ağ kurulacaktır (internet bağlantısı olmayacaktır)
- Yarışmacılar ethernet kablosu ile bağlanacaklardır
- Her oturumda, her takımdan aynı anda 3 yarışmacının yarışma alanına girişine izin verilecektir
- Yarışma esnasında bir takımın başka bir takıma yardımcı olmasına izin verilmemektedir

### 3.1. Test Oturumu

- Süre: 75 dakika
- Amaç: Donanım kurulumlarını yapmak
- Test videosu: 2 dakikalık (900 video karesi)
- Test oturumunda yollanan sonuçların puanlandırmada etkisi olmayacaktır

### 3.2. Yarışma Oturumları

- 4 yarışma oturumu yapılacaktır
- Her oturumun toplam süresi: 75 dakika
- İlk 15 dakika: Hazırlık
- Sonraki 60 dakika: Yarışma
- Her oturumda 2250 video karesi verilecektir
- Her oturumun bir teması olabilir (Güneşli, Zorlu Hava Şartları, Akşam, Deniz Üstü vb.)

---

## 4. TEKNİK SUNUM

- Yarışmacı takımlardan yarışma oturumları esnasında bir sunum yapmaları beklenmektedir
- Her takımdan bir adet İletişim Sorumlusu sunumu yapmak ile görevlendirilmelidir
- Sunum süresi: Takım başına maksimum 5 dakika
- Sunumlar 3 kişiden oluşan bir hakem heyetine sunulacaktır
- Hazırlanan sunumlar t3kys.com adresine yollanarak teslim edilmelidir

---

## 5. RAPORLAMA

Yarışmacı takımlardan iki ayrı doküman yazmaları beklenmektedir:

### 5.1. Ön Tasarım Raporu

- Şablon en geç 22/04/2026 tarihinde teknofest.org sitesinde paylaşılacaktır
- Yarışmaya katılım için Ön Tasarım Raporunu teslim etmek zorunludur
- Yarışma sonuçlarının belirlenmesinde Ön Tasarım Raporunun bir etkisi bulunmamaktadır
- Son teslim tarihi: 22/04/2026

### 5.2. Final Tasarım Raporu

- Şablon en geç Ağustos 2026 tarihinde teknofest.org sitesinde paylaşılacaktır
- Final Tasarım Raporu puanı genel yarışma puanının **%5'ini** oluşturmaktadır
- Bir yarışma takımının yarışmada dereceye girebilmesi için Final Tasarım Raporu'nu teslim etmesi zorunludur

---

## 6. ÇEVRİMİÇİ YARIŞMA SİMÜLASYONU

- Ön Tasarım Raporu değerlendirmelerinden sonra bir ön eleme yarışması yapılacaktır
- Çevrimiçi Yarışma Simülasyonu'nda yarışmacılardan geliştirdikleri modeller ile çevrimiçi ortamda paylaşılan karelerdeki nesneleri tespit etmeleri ve hava aracının pozisyonunu kestirmeleri beklenmektedir
- Belirlenen başarı kriterinin altında kalan ve sunucuya hiç bağlanmayan takımlar bir sonraki aşamaya geçemeyecektir

---

## 7. TAKIMLARIN YAZILIM VE DONANIM ÖZELLİKLERİ

- Her takım kendi yazılım ve donanım sisteminden sorumludur
- İstenilen işletim sistemi kullanılabilir
- Takımlar istedikleri platformda ve programlama dillerinde geliştirme yapabilir
- Yarışmacılardan saniyede 1 görüntü karesi işleyebilecek donanıma sahip olmaları yeterli olacaktır
- Algoritmanın çalışma hızı bir puanlandırma kriteri değildir

---

## 8. YARIŞMA SIRASINDA SUNUCU İLE BAĞLANTI

### Sunucu Bağlantı Bilgileri:

- Yarışma sunucusu adresi: Örn. http://127.0.0.25:5000
- Haberleşme: API mantığı ile JSON formatında
- Her takım tek bir IP adresi ile bağlanmalıdır

### Görüntü Karesi JSON Formatı:

```json
{
  "url": "video_karesi_id_url",
  "image_url": "video_karesi_gorsel_url",
  "video_name": "video_adi",
  "session": "oturum_url",
  "translation_x": "x_degeri",
  "translation_y": "y_degeri",
  "translation_z": "z_degeri",
  "gps_health_status": "saglik_degeri"
}
```

### Sonuç Gönderme JSON Formatı:

```json
{
  "id": "tahmin_id",
  "user": "kullanici_url",
  "frame": "video_karesi_url",
  "detected_objects": [
    {
      "cls": "0",  // Sınıf: "0", "1", "2", "3"
      "landing_status": "-1",  // İniş durumu: "-1", "0", "1"
      "motion_status": "1",  // Hareket durumu: "-1", "0", "1"
      "top_left_x": "x_koordinati",
      "top_left_y": "y_koordinati",
      "bottom_right_x": "x_koordinati",
      "bottom_right_y": "y_koordinati"
    }
  ],
  "detected_translations": [
    {
      "translation_x": "x_degeri",
      "translation_y": "y_degeri",
      "translation_z": "z_degeri"
    }
  ],
  "detected_undefined_objects": [
    {
      "object_id": "nesne_id",
      "top_left_x": "x_koordinati",
      "top_left_y": "y_koordinati",
      "bottom_right_x": "x_koordinati",
      "bottom_right_y": "y_koordinati"
    }
  ]
}
```

---

## 9. PUANLAMA

### Genel Yarışma Puanlandırması:

| Puan Türü | Puan Oranı |
|-----------|------------|
| Birinci Görev | %25 |
| İkinci Görev | %40 |
| Üçüncü Görev | %25 |
| Final Tasarım Raporu | %5 |
| Yarışma Sunumu | %5 |
| **Toplam Puan** | **%100** |

### 9.1. Birinci Görev Puanlama Kriteri

- Nesne tespitinin çalışma performansı **mAP** (mean Average Precision) değerine göre belirlenecektir
- mAP, **IoU** (Intersection Over Union) değeri üzerinden hesaplanır
- IoU eşik değeri: **0.5**

**IoU Formülü:**
```
IoU = (Gerçek Referans Çıktı ∩ Tahmin Edilen Çıktı) / (Gerçek Referans Çıktı ∪ Tahmin Edilen Çıktü)
```

**Örnek Puanlama Durumları:**

| Örnek | Gerçek Sınıf | Tespit Edilen Sınıf | IoU | İniş Değeri | Sonuç |
|-------|--------------|---------------------|-----|-------------|-------|
| 1 | İnsan | İnsan | 0.63 | Doğru | AP artışı |
| 2 | İnsan | Taşıt | 0.66 | Doğru | AP düşüşü |
| 3 | İnsan | İnsan | 0.42 | Doğru | AP düşüşü (IoU < 0.5) |
| 4 | Taşıt | Taşıt | 0.85, 0.61, 0.54 | -1 | AP düşüşü (çok tespit) |
| 5 | UAP | UAP | 0.91 | Yanlış | AP düşüşü |
| 6 | Taşıt | Tespit yok | - | - | AP düşüşü |

### 9.2. İkinci Görev Puanlama Kriteri

- Hava aracının referans pozisyon bilgisi ile yarışmacıların kestirdiği pozisyon bilgisi arasındaki ortalama hata kullanılarak puanlandırma yapılacaktır

**Ortalama Hata Hesaplama Formülü:**
```
E = (1/N) × Σ√((x̂ᵢ - xᵢ)² + (ŷᵢ - yᵢ)² + (ẑᵢ - zᵢ)²)
```

- x̂ᵢ, ŷᵢ, ẑᵢ: Yarışmacının i. görsel için yolladığı pozisyon kestirimi
- xᵢ, yᵢ, zᵢ: Hava aracının mutlak doğru pozisyon bilgisi

### 9.3. Üçüncü Görev Puanlama Kriteri

- Birinci görevdeki puan hesaplama yöntemi (mAP) kullanılacaktır
- Detaylı bilgilendirme şartname revizyonlarında verilecektir

---

## 10. YARIŞMA GİTHUB ve GOOGLE GROUPS SAYFALARI

### Github Proje Deposu:
- Yarışma boyunca kullanılacak kod blokları, örnek veri setleri ve diğer teknik materyaller Github üzerinden paylaşılacaktır

### Google Groups Platformu:
- Takımlar arası bilgi alışverişi için Google Groups tartışma platformu oluşturulmuştur
- Duyurular ve sıkça sorulan sorular bu platform üzerinden paylaşılacaktır

---

## 11. YARIŞMA SONUÇLARININ DUYURULMASI VE ÖDÜLLENDİRME

- Her yarışma oturumu başlarken, yarışmacıların bir önceki aldıkları puana göre sıralamaları bildirilecektir
- Tüm yarışma oturumlarının tamamlanmasının ardından, genel yarışma puanı hesaplanacaktır
- Yarışmada dereceye giren takımlar TEKNOFEST'in son gününde kürsüye çıkarak ödüllerini alacaklardır

---

*Bu teknik şartname TEKNOFEST Havacılıkta Yapay Zeka Yarışması için hazırlanmıştır.*
