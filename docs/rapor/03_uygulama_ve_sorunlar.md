# 3. UYGULAMA VE KARŞILAŞILAN SORUNLAR

Bu bölüm, tasarımın uygulanması sırasında ortaya çıkan uyumsuzlukları, her birinin kök
nedenini ve uygulanan çözümü anlatmaktadır. Buradaki sorunların tamamı gerçek koşularda
karşılaşılmış, tarih ve dosya bilgisiyle kayda geçirilmiştir; hiçbiri örnek amaçlı
üretilmemiştir.

Bölümün yazılış biçimi bilinçlidir. Karşılaşılan hataların çoğu, **sessizce yanlış sonuç
üreten** türdendir: program çökmez, bir sayı üretir, ve o sayı yanlıştır. Bu tür hatalar
ancak sonuç beklentiyle çeliştiğinde fark edilir; dolayısıyla her birinin nasıl fark
edildiği de anlatılmıştır.

## 3.1 SESSİZ HATALAR

### 3.1.1 Değerlendirme Rastgele Ağırlık Ölçüyordu

**Belirti.** Eğitilen modellerin doğrulama mIoU değerleri beklenenin çok altındaydı
(16,48 / 20,53 / 26,54 gibi). Eğitim logları çok daha yüksek değerler gösteriyordu.

**Kök neden.** Değerlendirme hattı modeli `Runner.from_cfg()` ile kuruyordu. Bu çağrı
yapılandırmadaki `load_from` alanını **okumaz**; ağırlık yükleme yalnızca
`Runner.train()`, `Runner.val()` ve `Runner.test()` içinde gerçekleşir. Dolayısıyla
değerlendirme, eğitilmiş ağırlıkları değil **rastgele başlatılmış ağırlıkları** ölçüyordu.

**Çözüm.** Ağırlık yükleme açık bir adıma alındı (`load_evaluation_weights()`) ve
yüklemenin gerçekleştiği, yüklenen dosyanın sha256 özeti kayda yazılarak doğrulandı. Söz
konusu mIoU değerleri geçersiz sayılmış ve bu raporda hiçbir yerde kullanılmamıştır.

**Ders.** Bir kütüphanenin bir işi yaptığını varsaymak yeterli değildir; yaptığının
gözlenebilir bir kanıtı olmalıdır.

### 3.1.2 ONNX Parite Kontrolü Geçmiyordu

**Belirti.** PyTorch ve ONNX çıktıları arasındaki sayısal fark, kabul eşiğinin üzerinde
kalıyordu.

**Kök neden.** Karşılaştırma PyTorch tarafında CUDA üzerinde TF32 hassasiyetiyle, ONNX
tarafında CPU üzerinde FP32 ile yapılıyordu. İki farklı hassasiyet karşılaştırılıyordu.

**Çözüm.** Parite kontrolü her iki taraf da CPU'da FP32 çalışacak biçimde yeniden
kuruldu; ayrıca sınıf haritası eşdeğerliği ayrı bir ölçüt olarak eklendi.

### 3.1.3 Hiperparametre Aramasının Erken Sonlanması

**Belirti.** Optimizasyon çalışması, bütçesi dolmadan tamamlanmış sayılıyordu.

**Kök neden.** Budanan (pruned) denemeler "tamamlanmış" kabul ediliyordu; oysa budanan bir
deneme sonuç üretmez.

**Çözüm.** Bütçe muhasebesi yalnızca gerçekten tamamlanan denemeleri sayacak biçimde
düzeltildi (`complete_trials_within_budget`).

### 3.1.4 Hata Günlüğü Paketi Sessizce Üretilmiyordu

**Belirti.** Hata durumunda üretilmesi gereken günlük paketi bazen boş çıkıyordu ve bu
durum hata vermiyordu.

**Kök neden.** ZIP biçimi 1980 öncesi zaman damgası taşıyamaz. Bazı dosyalar bu tarihten
önceki bir zaman damgasıyla oluştuğu için paketleme sessizce başarısız oluyordu.

**Çözüm.** Zaman damgaları paketleme öncesinde geçerli aralığa kırpıldı ve paketin boş
olmadığı doğrulandı.

### 3.1.5 AUROC Hesabı Gerçek Çalışma Zamanında Çöküyordu

**Belirti.** Open-set metrikleri yerel testte çalışıyor, gerçek çalışma zamanında
`AttributeError` veriyordu.

**Kök neden.** Kod `numpy.trapezoid` kullanıyordu; bu ad NumPy 2.0 ile gelmiştir, dağıtım
çalışma zamanı ise NumPy 1.26.4 üzerindedir. Eski ad olan `numpy.trapz` ise NumPy 2.4 ile
kaldırılmıştır — yani iki addan hiçbiri bütün sürümlerde güvenli değildir.

**Çözüm.** Yamuk kuralı doğrudan yazıldı. Üç satırlık bir hesap için sürüme bağlı bir ada
güvenmenin gerekçesi yoktur.

### 3.1.6 Eşik Politikası Hesabı 90 Dakika Sürüyordu

**Belirti.** Open-set eşik politikalarının hesaplanması 1,2 milyon piksel üzerinde yaklaşık
90 dakika sürüyordu.

**Kök neden.** Her aday eşik için bütün piksellerin yeniden taranması, yani O(n²)
karmaşıklık.

**Çözüm.** Sıralı skorlar üzerinde kümülatif sayımlarla vektörleştirildi; süre 5 dakika 45
saniyeye indi. Doğruluğun korunduğu, eski yavaş uygulama test içinde saklanarak ve iki
uygulamanın çıktıları karşılaştırılarak garanti altına alındı.

## 3.2 ÖLÇÜM PROTOKOLÜ HATALARI

### 3.2.1 Uç Cihaz Ölçümü On Dakikayı Boşa Harcıyordu

**Belirti.** 600 saniyelik sürdürülen yük ölçümü tamamlandıktan sonra, telemetri kaydının
alınamadığı anlaşılıyordu; ölçüm baştan yapılmak zorunda kalıyordu.

**Kök neden.** Telemetrinin çalışıp çalışmadığı kontrolü, ölçümden **sonra** yapılıyordu.

**Çözüm.** Kontrol ön koşula alındı: telemetri okunamıyorsa ölçüm hiç başlamaz. Bu
davranış bir regresyon testiyle sabitlenmiştir.

**Ders.** Uzun süren bir ölçümün ön koşulları, ölçüm başlamadan doğrulanmalıdır.

### 3.2.2 Optimizasyon Kararı Yanlış Makinede Verildi

**Belirti.** Bağlantılı bileşen etiketleme için iki uygulama karşılaştırıldı: klasik
genişlik-öncelikli arama ve dizi tabanlı vektörleştirme. Geliştirme makinesinde (Apple
Silicon) arama **1,4–2,5 kat** hızlıydı, bu nedenle vektörleştirme yazıldıktan sonra geri
alındı.

**Kök neden.** Ölçüm hedef donanımda yapılmamıştı. ARM tabanlı Jetson işlemcisinde Python
yorumlayıcısı, NumPy'ye göre çok daha yavaş çalışır; bu da iki uygulama arasındaki dengeyi
tersine çevirir.

**Çözüm.** Karşılaştırma hedef cihazda tekrarlandı. Jetson üzerinde vektörleştirme
**1,59 kat** hızlı çıktı (220 gerçek 64×128 maske üzerinde 3,154 ms → 1,989 ms) ve karar
geri alındı.

> **Bu, raporun en genellenebilir dersidir: uç cihaz optimizasyon kararları geliştirme
> makinesinde alınamaz.** Aynı iki uygulama, aynı girdilerle, iki farklı makinede zıt
> sonuç vermiştir.

### 3.2.3 Belgelenen Komut Çalışmıyordu

**Belirti.** FP16 dağıtım sadakati ölçümü, 239 MB'lık değerlendirme verisi cihaza
aktarıldıktan sonra `ModuleNotFoundError` ile durdu.

**Kök neden.** Bir betik `python scripts/jetson/evaluate_engine.py` biçiminde
çalıştırıldığında Python **o dizini** içe aktarma yoluna ekler, depo kökünü değil. Betiğin
kardeş modülden aldığı çalışma zamanı sınıfı bu nedenle bulunamıyordu. Kılavuz tam olarak
bu çağrıyı belgeliyordu.

**Çözüm.** Betikler depo kökünü kendileri içe aktarma yoluna ekler hâle getirildi.
Sorunun fark edilmemiş olmasının nedeni, test paketinin bu modülleri paket yolu üzerinden
içe aktarması ve orada sorunun görünmemesiydi; bu nedenle **belgelenen çağrının kendisini**
alt süreç olarak çalıştıran bir test yazıldı ve `scripts/` altındaki 69 betiğin tamamına
uygulandı.

## 3.3 SONUÇLARIN GERİ ÇEKİLMESİ

İki bulgu, sonradan yapılan ölçümlerle çürütülmüş ve geri çekilmiştir. Her ikisi de bu
raporda yer almaktadır.

**Birincisi**, doğruluk ile open-set başarımı arasında güçlü bir ters ilişki
bulunduğu (ρ = −0,90) iddiasıydı. Bu korelasyon **yayımlanmış** mIoU değerleriyle
hesaplanmıştı. Kendi dağıtım koşulumuzda ölçülen değerlerle aynı hesap ρ = +0,30
vermektedir; yani ilişki yoktur. Bulunan şey mimarilerin bir özelliği değil, yayımlanmış
sıralamanın dağıtım koşuluna taşınmamasıdır — ki bu zaten raporun birinci bulgusudur.

**İkincisi**, bir mimarinin uçtan uca maliyetinin "entegrasyon artığı" olduğu ve
giderilebileceği varsayımıydı. Ölçüldüğünde maliyetin gerçek doğruluk taşıdığı görülmüştür
(Bölüm 5.7).

## 3.4 PROJE, RİSK VE DEĞİŞİKLİK YÖNETİMİ

Proje boyunca üç mekanizma kullanılmıştır:

**Karar kaydı.** Her materyal değişiklik; tarihi, dokunulan dosyalar, yapılan doğrulama ve
gerekçesiyle birlikte **ekle-sadece** bir günlüğe yazılmıştır. Kayıt geriye dönük
düzenlenmez; bir karar değiştiğinde eski satır silinmez, yeni satır eklenir. Geri çekilen
iki sonuç bu günlükte kendi düzeltmeleriyle birlikte durmaktadır.

**Kapsam kısıtları.** Bazı işlemler, sonuçların geçerliliğini bozabileceği için kod
düzeyinde kısıtlanmıştır. Örneğin sıcaklık ölçekleme yalnızca kalibrasyon rolündeki veride
yapılabilir ve mühürlü nihai test verisinin açılması otomatik bir süreçle tetiklenemez.

**Sürekli doğrulama.** 790 birim testi her değişiklikte çalıştırılmıştır. Testlerin bir
bölümü doğrudan karşılaşılmış hatalara karşılık gelir ve hatanın kendisini değil, hatanın
**koşulunu** sabitler.

## 3.5 YENİDEN ÜRETİLEBİLİRLİK

Çalışmanın yeniden üretilebilirliği dört düzeyde sağlanmıştır:

1. **Rastgelelik.** Bütün rastgele işlemler sabit tohum (20260728) kullanır ve tohum,
   ölçüm kaydının içine yazılır.
2. **Veri.** Değerlendirme kümeleri dondurulmuş manifestlerle tanımlanmıştır; hangi
   karelerin kullanıldığı kayıttan geri izlenebilir.
3. **Model.** Her ölçüm kaydı, kullanılan ağırlık dosyasının sha256 özetini taşır. Bir
   sayının hangi ağırlıkla üretildiği kayıttan doğrulanabilir.
4. **Çalışma zamanı.** Bağımlılıklar sürüm bazında sabitlenmiş, uç cihaz motoru
   `engine_sha256` ile kimliklendirilmiştir.

Buna ek olarak, raporda geçen dört ondalıklı her sayının bir ölçüm kaydında karşılığının
bulunup bulunmadığı otomatik olarak denetlenmektedir. Denetim, elle yazılmış veya bayat
kalmış bir sayıyı test aşamasında yakalar; nitekim yazım sırasında iki tutarsızlığı bu
denetim ortaya çıkarmıştır.
