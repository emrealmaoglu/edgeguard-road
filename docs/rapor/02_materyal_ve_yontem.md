# 2. MATERYAL VE YÖNTEM

Bu bölümde çalışmada kullanılan veri setleri, karşılaştırılan mimariler, sistem mimarisi,
ölçüm protokolü ve metrikler tanımlanmaktadır. Bölümün sonunda, bütün doğruluk sayılarının
dayandığı veri bütünlüğü varsayımının nasıl denetlendiği anlatılmaktadır.

## 2.1 VERİ SETLERİ

Çalışmada dört veri seti kullanılmıştır. Her birinin rolü baştan sabitlenmiş ve rol
değişimi yapılmamıştır; bir veri setinin hem model seçiminde hem nihai raporlamada
kullanılması, sonuçları iyimser gösteren yaygın bir metodolojik hatadır.

Çizelge 2.1. Kullanılan veri setleri ve rolleri.

| Veri seti | Rol | Boyut | Kaynak |
|---|---|---|---|
| Cityscapes val | Doğruluk, sınıf bazlı IoU, kalibrasyon | 500 kare | [7] |
| ACDC val | Gerçek olumsuz koşul dayanıklılığı | 406 kare | [5] |
| RoadAnomaly | Open-set yol tehlikesi | 60 kare, piksel etiketli | [6] |
| Cityscapes demoVideo | Zamansal tutarlılık, nitel gösterim | 180 ardışık kare | [7] |
| Cityscapes + IDD20K | Kendi sınırlı bütçeli eğitimimiz | dondurulmuş manifest | [7], [8] |

**Cityscapes** [7], 50 şehirden toplanmış 5.000 ince anotasyonlu görüntü içerir ve 19
sınıflık etiket şeması bu çalışmadaki bütün ontolojinin temelidir. Doğrulama kümesinin
tamamı (500 kare) doğruluk ve kalibrasyon ölçümlerinde kullanılmıştır.

**ACDC** [5], sis, gece, yağmur ve kar koşullarına eşit dağılmış, piksel düzeyinde
anotasyonlu bir olumsuz koşul veri setidir. Bu çalışmada kullanılma nedeni, sentetik
bozulmaların gerçek olumsuz koşulların yerini tutmamasıdır; Bölüm 5'te bu iki ölçüm
karşılaştırmalı olarak verilmiştir.

**RoadAnomaly** [6], yol üzerinde bulunan ve eğitim dağılımında yer almayan nesneleri
piksel düzeyinde etiketler. Bu çalışmadaki open-set değerlendirmesinin tek gerçek veri
kaynağıdır; karşılaştırılan modellerin hiçbiri bu veriyle eğitilmemiş veya ince ayara tabi
tutulmamıştır.

**IDD20K** [8], yapılandırılmamış yollarda toplanmıştır. Varlık nedeni, Cityscapes'in
dayandığı varsayımların — şeritli yol, az sayıda iyi tanımlı nesne sınıfı, trafik
kurallarına uyum — Hindistan yollarında geçerli olmamasıdır. Kendi çok-domainli eğitim
kararımızın gerekçesi budur.

### 2.1.1 FAIR İlkelerine Uygunluk

Veri yönetimi, FAIR (Findable, Accessible, Interoperable, Reusable) ilkelerine göre
kurgulanmıştır:

- **Bulunabilirlik.** Her veri seti, içindeki her örneğin yolunu ve sha256 özetini taşıyan
  dondurulmuş bir manifestle (`*.frozen.json`) tanımlanmıştır. Bir ölçümün hangi kareler
  üzerinde yapıldığı manifestten geriye doğru izlenebilir.
- **Erişilebilirlik.** Dört veri setinin dördü de resmî kaynağından, kendi lisans
  koşullarıyla edinilmiştir. Hiçbir veri yeniden dağıtılmamaktadır; depoda yalnızca
  manifestler ve ölçüm kayıtları bulunur.
- **Birlikte çalışabilirlik.** Etiket şemaları ortak bir 19 sınıflık `trainId` uzayına
  eşlenmiştir. Cityscapes tarafında eşleme, veri setinin kendi resmî
  `cityscapesscripts` etiket tablosundan alınmıştır; ACDC'nin kendi `trainId` etiketleriyle
  bu eşlemenin tutarlılığı 40 kare üzerinde ayrıca doğrulanmıştır.
- **Yeniden kullanılabilirlik.** Bütün ölçüm kayıtları makine tarafından okunabilir JSON
  biçimindedir ve kendi şema sürümünü, üretim zamanını, kullanılan model ağırlığının
  sha256 özetini ve rastgelelik tohumunu taşır.

Rol ayrımı `train_fit` / `train_select` / `train_calibration` olarak manifest düzeyinde
tanımlanmıştır. Sıcaklık ölçekleme yalnızca `train_calibration` rolündeki veride yapılır;
bu kısıt kod düzeyinde uygulanır ve aşılamaz.

## 2.2 KARŞILAŞTIRILAN MİMARİLER

Çalışmada beş gerçek zamanlı semantik segmentasyon mimarisi karşılaştırılmıştır. Bu
mimariler, gerçek zamanlı segmentasyon literatüründe farklı tasarım felsefelerini temsil
ettikleri için seçilmiştir.

Çizelge 2.2. Karşılaştırılan mimariler ve tasarım yaklaşımları.

| Mimari | Tasarım yaklaşımı | Kaynak |
|---|---|---|
| PIDNet-M / PIDNet-S | Üç dallı yapı; PID denetleyici benzeşimiyle detay ve bağlam füzyonu | [1] |
| DDRNet-23-slim | İki dallı omurga + Deep Aggregation Pyramid Pooling Module | [3] |
| SegFormer-B0 | Hiyerarşik transformer kodlayıcı + hafif MLP kod çözücü, konum kodlaması yok | [2] |
| BiSeNetV2 | Detail Branch + Semantic Branch + Guided Aggregation Layer | [4] |

Ölçümler, mmsegmentation model zoo'sunda yayımlanmış, Cityscapes üzerinde 120.000–160.000
adım eğitilmiş referans checkpoint'lerle yapılmıştır (Apache-2.0 lisansı). Bu tercihin
nedeni, beş mimariyi sınırlı bir bütçede yeniden eğitmenin karşılaştırmayı eğitim
bütçesinin karşılaştırmasına dönüştürecek olmasıdır.

Bunun yanında, Cityscapes ve IDD20K karışımıyla kendi sınırlı bütçeli eğitimimiz de
yapılmıştır (2.500 adım, batch 4, 512×1024 kırpma, bf16, ImageNet başlangıcı, tohum
20260728). Bu eğitim yayımlanmış checkpoint'lerin yaklaşık **%0,7'si kadar** örnek
görmüştür.

> **Bu iki hat rapor boyunca ayrı tutulmuş, hiçbir yerde karıştırılmamıştır.** Kendi
> eğitimimizin sonuçları çok-domainli genelleme denemesi olarak, referans
> checkpoint'lerinki ise mimari karşılaştırması olarak sunulur.

## 2.3 SİSTEM MİMARİSİ

Sistem, tek bir kamera karesinden başlayıp sıralanmış risk bölgelerine ulaşan tek yönlü
bir işleme hattıdır. Şekil 2.1 bu akışı ve her aşamanın hangi çıktıyı ürettiğini
göstermektedir.

**[Şekil 2.1: D1_system_architecture.pdf]**

Şekil 2.1. Görüntüden risk sıralamasına sinyal yolu.

Hat üç bölümden oluşur. **Birinci bölüm** girdi hazırlığıdır: 2048×1024 kamera karesi
dağıtım çözünürlüğü olan 512×1024'e ölçeklenir ve normalize edilir. **İkinci bölüm**
TensorRT FP16 motorudur ve 19 kanallı logit haritası üretir. **Üçüncü bölüm** tamamen CPU
tarafındadır: logitlerden semantik maske (argmax), güven ve entropi haritaları, ardından
yol maskesi, ego koridoru, bağlantılı bölgeler ve güvenilmez piksel maskesi türetilir.
Bölüm 4'te gösterileceği üzere, kare bütçesinin büyük çoğunluğu bu üçüncü bölümde
harcanmaktadır.

Risk sıralaması, yedi özellikli ağırlıklı bir füzyonla üretilir: anomali skoru, bileşen
alanı, görüntüdeki konum, yol ile komşuluk, koridora göreli yakınlık, dedektör örtüşmesi ve
zamansal kalıcılık. Ölçülemeyen bir özellik **sıfır değerle değil sıfır ağırlıkla**
dışlanır; sıfır değer verilmesi, ölçülmemiş bir sinyali "risk yok" gibi göstererek bütün
skorları aşağı çekerdi.

## 2.4 ÖLÇÜM PROTOKOLÜ

Şekil 2.2, hangi verinin hangi soruyu nerede ölçerek yanıtladığını göstermektedir.

**[Şekil 2.2: D2_measurement_protocol.pdf]**

Şekil 2.2. Ölçüm protokolü: hangi veri hangi soruyu nerede yanıtlıyor.

Değerlendirme üç ayrı eksende yürütülmüştür ve Şekil 2.3'te görüldüğü gibi bu eksenlerin
kazananları aynı model değildir.

**[Şekil 2.3: D3_three_axis_framework.pdf]**

Şekil 2.3. Üç eksenli değerlendirme çerçevesi.

### 2.4.1 Donanım ve Çalışma Zamanı

Uç cihaz ölçümleri NVIDIA Jetson Orin Nano Super üzerinde yapılmıştır. Bu modül 1024 CUDA
ve 32 Tensor çekirdeği, 8 GB 128-bit LPDDR5 bellek (102 GB/s) ve 6 çekirdekli Arm
Cortex-A78AE işlemci içerir; 7 W, 15 W ve 25 W güç modlarını destekler [12].

Bütün ölçümler **25 W** modunda alınmıştır. Çalışma zamanı TensorRT 10.3.0 FP16, JetPack 6
(L4T 36, aarch64)'dır.

### 2.4.2 Sürdürülen Yük Protokolü

Uç cihaz ölçümlerinde tek kare gecikmesi yerine **sürdürülen yük** ölçülmüştür. Gerekçesi
şudur: kısa bir ölçüm, cihazın henüz ısınmadığı ve saat frekanslarının düşürülmediği bir
pencereden alınır ve gerçek dağıtım koşulunu temsil etmez.

Protokol her model için aynıdır: 200 kare ısınma, ardından **600 saniye kesintisiz yük**,
bu süre boyunca saniyede bir örnekle tam telemetri kaydı (güç, bellek, sıcaklık). Ölçüm
karelerinin diskten okunması da süreye dâhildir; gerçek bir dağıtımda bu maliyet vardır.

Termal davranış, saat frekansı düşürme uyarı bayrağına değil doğrudan **sıcaklık
serisine** bakılarak değerlendirilmiştir. Bunun nedeni pratiktir: `tegrastats` stok
çıktısında böyle bir uyarı metni bulunmamaktadır. Literatürde de kısıtlamanın gerçekleştiği
eşik doğrudan sıcaklık değerlerinden okunmaktadır [14].

## 2.5 METRİKLER

Çizelge 2.3. Kullanılan metrikler ve ölçtükleri özellik.

| Eksen | Metrik | Not |
|---|---|---|
| Doğruluk | mIoU, sınıf bazlı IoU, piksel doğruluğu | 19 sınıf makro ortalama |
| Doğruluk | Bileşen kapsama, bileşen IoU, parçalanma | Nesne bütünlüğü; piksel metriklerinin göremediği |
| Doğruluk | Yol IoU, sınır F1, yanlış-sürülebilir oran | Sürülebilir alan |
| Kalibrasyon | ECE, sınıf-bazlı ECE, güvenilirlik eğrisi | 15 kutu |
| Open-set | AUROC, AP, FPR95 | Eşikten bağımsız |
| Zamansal | İz ömrü, titreme (kategori değişimi) | 150 ardışık kare |
| Uç maliyet | Gecikme (medyan/p95/p99), sürdürülen FPS | Motor ve uçtan uca ayrı |
| Uç maliyet | Ortalama/tepe güç, joule/kare, tepe RAM, sıcaklık | Modül içi sensörler |
| İstatistik | Eşleştirilmiş fark, %95 bootstrap güven aralığı | 1.000 yeniden örnekleme, tohum sabitli |

Sınıf-bazlı ECE'nin ayrıca hesaplanma nedeni Bölüm 5.4'te ölçümle gösterilmiştir:
havuzlanmış tek bir ECE değeri, baskın sınıfların iyi kalibre kütlesi nedeniyle iyimser
çıkmaktadır.

### 2.5.1 Farkların İstatistiksel Değerlendirilmesi

Beş mimari karşılaştırıldığında, aradaki farkların gerçek mi yoksa hangi karelerin
doğrulama kümesine düştüğüyle ilgili bir rastlantı mı olduğu sorusu ortaya çıkar. Bu
çalışmada bu soru boş bırakılmamıştır.

Modeller **aynı karelerde** puanlanmış, kare başına skorlar saklanmış ve her model çifti
için fark **eşleştirilmiş bootstrap** ile değerlendirilmiştir. Eşleştirme burada
belirleyicidir: kareler zorluk bakımından büyük fark gösterir ve bu değişkenlik modeller
arasında ortaktır; eşleştirilmemiş ortalamalar karşılaştırıldığında küçük farklar bu ortak
değişkenliğin içinde kaybolur.

Kayıtlarda hiçbir yerde "istatistiksel olarak anlamlı" ifadesi kullanılmamaktadır. Rapor
edilen şey, güven aralığının sıfırı içerip içermediğidir.

## 2.6 VERİ BÜTÜNLÜĞÜ: SPLIT SIZINTISI DENETİMİ

Bu raporda verilen bütün doğruluk sayıları, değerlendirme kümelerinin birbirinden bağımsız
olduğu varsayımına dayanır. Sürüş görüntüsü saniyede 15–30 kare hızında kaydedildiği için
saniyenin üçte biri arayla iki kare neredeyse aynıdır; bu tür kareler farklı kümelere
düşerse model segmentasyon yerine ezberlediği arka planı tanımayı öğrenir ve metrikler
yapay olarak yükselir.

Varsayım denetlenmiştir. Dört değerlendirme kümesinden eşit aralıklı 446 kare alınmış,
her kare 64 bitlik bir ortalama-hash ile özetlenmiş ve farklı yarıçaplarda yakın-kopya
çiftleri sayılmıştır.

Çizelge 2.4. Algısal yakın-kopya denetimi (446 kare, 64-bit ortalama-hash).

| Hamming yarıçapı | Splitler arası | cityscapes_val | demo_video | acdc_night | acdc_fog |
|---|---|---|---|---|---|
| d ≤ 0 | **0** | 0 | 36 | 0 | 2 |
| d ≤ 2 | **0** | 0 | 127 | 1 | 88 |
| d ≤ 4 | 7 | 2 | 293 | 12 | 581 |
| d ≤ 6 | 90 | 16 | 516 | 80 | 1.504 |

Splitler arası yakın-kopya sayısı d ≤ 2'de **sıfırdır**. Yarıçap büyütüldüğünde sayının
0 → 0 → 7 → 90 biçiminde artması, gerçek kopyaların varlığına değil, hash'in ayırt etme
gücünün tükenmesine işaret eder; gerçek bir kopya d = 0'da görünür ve yarıçap büyüdükçe
orada kalır. Cityscapes doğrulama kümesi kendi içinde de temizdir. demoVideo'nun d = 0'da
36 çift vermesi beklenen bir sonuçtur: o küme bitişik bir video dizisidir.

Denetim ayrıca yöntemsel bir sınır ortaya çıkarmıştır. ACDC sis kümesinde çift sayısı d = 0
ile d = 6 arasında 2'den 1.504'e çıkarken (752 kat), Cityscapes'te 0'dan 16'ya
çıkmaktadır. Sis, ortalama-hash'in dayandığı kontrastı yok etmektedir; dolayısıyla **sabit
bir algısal-hash eşiği farklı hava koşulları arasında taşınamaz.** Olumsuz koşul verisinde
yinelenen kayıt ayıklaması yapan çalışmaların bu sınırı hesaba katması gerekir.

Algısal hash bir kimlik kanıtı değildir; kayıt bunu `identity_proof: false` alanıyla
belirtir. Denetimin gösterdiği şey, iki karenin farklı sahneler olduğu değil, bu yarıçapta
yakın-kopya olmadıklarıdır.
