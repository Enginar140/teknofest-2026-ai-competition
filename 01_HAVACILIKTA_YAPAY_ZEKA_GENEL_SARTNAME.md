# HAVACILIKTA YAPAY ZEKA YARIŞMASI - GENEL ŞARTNAME

## VERSİYONLAR

| VERSİYON | TARİH | Açıklama |
|----------|-------|---------|
| V1.0 | 19.01.2025 | TEKNOFEST 2026 İlk Versiyon |
| V1.1 | 21.01.2026 | TEKNOFEST 2026 İkinci Versiyon |
| V1.1 | 20.02.2026 | Yarışma Son Başvuru Tarihi |

## TANIMLAR VE KISALTMALAR DİZİNİ

- **KYS:** TEKNOFEST Kurumsal Yönetim Sistemi
- **Takım Kaptanı:** Takımın organizasyonundan sorumlu olan ve süreçlerde liderlik görevini üstlenen kişi
- **Takım Danışmanı:** Her takım için en fazla bir (1) öğretmen/eğitmen/akademisyen
- **TEKNOFEST:** Havacılık, Uzay ve Teknoloji Festivali
- **T3 Vakfı:** Türkiye Teknoloji Takımı Vakfı
- **Yarışma Süreci:** Yarışma başvurularının alınmaya başladığı tarih ile final sonuçlarının açıklandığı tarih arasında geçen süre

---

## 1. YARIŞMANIN AMACI

Yapay Zeka teknolojilerinin geliştirilmesi ve üretimi, ülkemizin teknolojik geleceği ve rekabet gücü açısından stratejik bir öneme sahiptir. Bu kapsamda düzenlenen "Havacılıkta Yapay Zeka" Yarışması, havacılık sektöründe karşılaşılabilecek sorunlara yenilikçi çözümler sunmayı hedeflemektedir. Yarışma, bilgi birikimini artırmak, bu alanda yetkin insan kaynağı yetiştirmek ve ülkemizin havacılık ve teknoloji alanındaki bağımsızlığını güçlendirmek amacıyla hayata geçirilmiştir.

---

## 2. YARIŞMANIN KAPSAMI

Günümüzün hızla gelişen teknolojik ortamında, havacılık sektörü, yenilikçi teknolojilerin yaygınlaşması ve büyük ölçekli değişimlerle karşı karşıyadır. Otonom uçan araçlar, hava taksileri ve diğer inovatif çözümler, havacılığın geleceğini şekillendiren en önemli unsurlar arasında yer almaktadır.

### Birinci Görev: Nesne Tespiti
Hava araçlarının kamera görüntülerini işleyerek çevresel farkındalık sağlayan bir nesne tespit sistemi geliştirilmesi beklenmektedir. Bu sistem:
- İnsanları
- Taşıtları
- Uçan Araba Park (UAP) alanlarını
- Uçan Ambulans İniş (UAİ) alanlarını

tanımlamalıdır.

**Tespit Edilecek Nesne Türleri:**
- **Taşıt** (Hareket Durumu: Hareketli/Hareketsiz)
- **İnsan**
- **Uçan Araba Park (UAP) Alanı** (İniş Durumu: Uygun/Uygun Değil)
- **Uçan Ambulans İniş (UAİ) Alanı** (İniş Durumu: Uygun/Uygun Değil)

### İkinci Görev: Pozisyon Tespiti
Hava araçlarının uydu tabanlı konumlandırma sistemlerinin (GPS gibi) devre dışı kaldığı durumlarda da güvenli bir şekilde uçabilmesi için görsel veriye dayalı bir pozisyon kestirim sistemi geliştirilmesi beklenmektedir.

### Üçüncü Görev: Referans Obje Tespiti
Hava araçlarının daha önce tanımlanmamış yeni objeleri görsel veri üzerinden anlık olarak tanıma ve takip etme yeteneğini test etmektedir.

---

## 3. YARIŞMA BAŞVURUSU

Yarışmaya; Havacılık, Uzay ve Teknoloji Festivali TEKNOFEST resmî web sitesi üzerinden (www.teknofest.org) **28 Şubat 2026** tarihine kadar başvuru yapılmalıdır. 

**Takım Oluşumu:**
- Takımlar en az 2, en fazla 5 kişiden oluşmalıdır
- 1 kişi İletişim Sorumlusu olarak görevlendirilmelidir
- Takım içerisinde bir takım kaptanı bulunmalıdır

---

## 4. YARIŞMA DETAYLARI

### 4.1. Yarışma Sırasında Kullanılacak Donanımlar
- Tüm takımlar geliştirdikleri yazılımları kendi bilgisayarlarında çalıştıracaklardır
- Bilgisayarların ethernet bağlantı girişine sahip olması zorunludur
- Yarışma sırasında kullanılan bilgisayarların internete bağlı olması kesinlikle yasaktır

### 4.2. Tespit Edilecek Nesne Türleri

#### Sınıf ID Bilgileri:

| Sınıf | Sınıf ID | İniş Durumu Değerleri | Hareket Durumu Değerleri |
|-------|----------|----------------------|-------------------------|
| Taşıt | 0 | -1 | 0,1 |
| İnsan | 1 | -1 | -1 |
| Uçan Araba Park (UAP) Alanı | 2 | 0,1 | -1 |
| Uçan Ambulans İniş (UAİ) Alanı | 3 | 0,1 | -1 |

#### İniş Durumu ID Bilgileri:
- **0:** Uygun Değil
- **1:** Uygun
- **-1:** İniş Alanı Değil

#### Hareketlilik Durumu ID Bilgileri:
- **0:** Hareketsiz
- **1:** Hareketli
- **-1:** Taşıt Değil

### 4.3. Tespit Edilecek Pozisyon Bilgisi
- Video başlangıcını referans alarak X eksenindeki metre cinsinden değişim
- Video başlangıcını referans alarak Y eksenindeki metre cinsinden değişim
- Video başlangıcını referans alarak Z eksenindeki metre cinsinden değişim

### 4.4. Yarışma Oturumları
- Toplam 4 oturumdan oluşmaktadır
- Her oturumda 5 dakikalık (2250 görüntü karesi) video sağlanacaktır
- Her oturumdan önce 15 dakikalık hazırlık süresi verilecektir
- Yarışma süresi: 60 dakika

### 4.5. Eğitim, Test ve Yarışma Videoları
- Birinci görev için örnek görüntüler ve nesne verileri paylaşılacaktır
- İkinci görev için örnek pozisyon bilgisi paylaşılacaktır

---

## 5. PUANLANDIRMA

Puanlandırma detayları yarışma teknik şartnamesinde açıklanacaktır.

---

## 6. YARIŞMA TAKVİMİ

| Tarih | Açıklama |
|-------|----------|
| 16.02.2026 | Teknik Şartnamenin İlanı |
| 28.02.2026 | Yarışma Son Başvuru Tarihi |
| 10-28.03.2026 | Örnek Eğitim Videosunun (Etiketsiz) Teslimi |
| 22.04.2026 - 17:00 | Ön Tasarım Raporu Son Teslim Tarihi |
| 22.05.2026 | Ön Tasarım Raporu Sonuçlarına göre 1. Ön Elemeyi Geçen Takımların Açıklanması |
| 01-06.06.2026 | Takımlarla Soru-Cevap Toplantısı |
| 09.07.2026 | Çevrim İçi Yarışma Simülasyonu |
| 17.07.2026 | Çevrim İçi Yarışma Simülasyonunun Sonuçlarına göre 2. Ön Elemeyi Geçen Takımların Açıklanması |
| Ağustos-Eylül 2026 | Yarışma Finalleri |
| Ağustos-Eylül 2026 | Final Tasarım Raporu Son Teslim Tarihi |
| 30 Eylül-4 Ekim 2026 | TEKNOFEST |

---

## 7. ÖDÜLLER

### Para Ödülleri:

| Derece | Ödül Miktarı | Danışman Ödülü |
|--------|--------------|----------------|
| BİRİNCİLİK | 250.000,00 ₺ | 15.000,00 ₺ |
| İKİNCİLİK | 225.000,00 ₺ | 12.000,00 ₺ |
| ÜÇÜNCÜLÜK | 200.000,00 ₺ | 10.000,00 ₺ |

### Prestij Ödülleri:
- **En İyi Sunum Ödülü:** Sunum becerilerini en iyi yansıtan takıma verilir (maddi karşılığı yoktur)
- **Yenilikçi Yazılım Ödülü:** Üçüncü görevi en iyi tamamlayan takıma verilir (maddi karşılığı yoktur)

### Başarı Kriteri:
Bir takımın başarılı sayılabilmesi için tüm oturumlarda elde ettiği puan yüzdelerinin ortalamasının %70'i geçmesi gerekmektedir.

---

## 8. YARIŞMAYA KATILIM KOŞULLARI

### 8.1. Yarışmaya Katılma Koşulları ve Detayları
- Türkiye ve yurt dışında öğrenim gören lise, üniversite öğrencileri ve mezunları takım halinde katılabilir
- Lise mezunu üyelerin mezuniyet tarihinden itibaren en fazla 3 yıl geçme şartı aranmaktadır
- Mezun kategorisi lise mezunu ve üniversite mezunlarını kapsamaktadır

### 8.2. Takım Oluşturma
- Takımlar en az 2, en fazla 5 kişiden oluşmalıdır
- Finalist olan takımlar final yarışması sırasında her takımdan en fazla 3 yarışmacı bulunabilir
- Çevrim İçi Yarışma Simülasyonu aşamasında başarılı olan takımlardan her okuldan en fazla 3 takım finale kalacaktır
- Yarışmaya bireysel katılım sağlanamaz

### 8.3. Danışman Yükümlülükleri
- Lise seviyesindeki takımlar bir danışman almak zorundadır
- Üniversite ve üzeri seviyesinde yarışacak takımların danışman alma zorunluluğu bulunmamaktadır
- Final aşamasına kalan projelerde lise seviyesinde takımların danışmanları ile alanda bulunmaları zorunludur

---

## 9. İLETİŞİM

Yarışma hakkında sorular için TEKNOFEST web sitesinde Havacılıkta Yapay Zeka Yarışması sayfasından yarışmanın grubuna katılabilirsiniz.

Organizasyonel sorular: iletisim@teknofest.org

---

## 10. GENEL KURALLAR

Yarışma kapsamında geçerli olan Genel Kurallar kitapçığına TEKNOFEST web sitesinden ulaşılabilir.

---

## 11. ETİK KURALLAR

Yarışma kapsamında geçerli olan Etik Kurallar kitapçığına TEKNOFEST web sitesinden ulaşılabilir.

---

## SORUMLULUK BEYANI

T3 Vakfı ve TEKNOFEST, yarışmacıların teslim etmiş olduğu herhangi bir üründen veya yarışmacıdan kaynaklanan herhangi bir yaralanma veya hasardan hiçbir şekilde sorumlu değildir. Yarışmacıların 3. kişilere verdiği zararlardan T3 Vakfı ve organizasyon yetkilileri sorumlu değildir.

Türkiye Teknoloji Takımı Vakfı işbu şartnamede her türlü değişiklik yapma hakkını saklı tutar.
