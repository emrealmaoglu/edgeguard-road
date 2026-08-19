# 5. BULGULAR VE TARTIŞMA

Bu bölüm ölçüm sonuçlarını dört eksende sunmakta ve bulguları tartışmaktadır. Bölümün
sonunda, çalışma sırasında geri çekilen iki sonuç ve raporun sınırları yer almaktadır.

## 5.1 YAYIMLANMIŞ SIRALAMA DAĞITIM SIRALAMASINI ÖNGÖRMÜYOR

Uç cihaza model seçen bir ekibin elindeki başlıca bilgi, model zoo'ların yayımladığı
doğruluk sıralamasıdır. Bu sıralamanın dağıtım koşuluna taşınıp taşınmadığı ölçülmüştür.

Şekil 5.1, yayımlanmış mIoU değerleri ile dağıtım çözünürlüğünde ölçülen değerleri
karşılaştırmaktadır.

**[Şekil 5.1: 01_published_vs_measured_miou.pdf]**

Şekil 5.1. Yayımlanmış ve ölçülen mIoU karşılaştırması.

Çizelge 5.1. Yayımlanmış ve dağıtım çözünürlüğünde ölçülen doğruluk.

| Mimari | Yayımlanmış mIoU | Ölçülen mIoU | Sıra değişimi |
|---|---|---|---|
| PIDNet-M | 80,22 | 0,6847 | 1 → 3 |
| PIDNet-S | 78,74 | 0,6768 | 2 → 4 |
| DDRNet-23-slim | 77,84 | 0,6850 | 3 → 2 |
| SegFormer-B0 | 76,54 | 0,6934 | 4 → 1 |
| BiSeNetV2 | 75,76 | 0,6602 | 5 → 5 |

İki sıralama arasındaki Spearman sıra korelasyonu **ρ = +0,10**'dur (n = 5). Yayımlanmış
sıralamada sonuncu olan SegFormer-B0, dağıtım çözünürlüğünde sınıf-ortalamalı mIoU'da öne
geçmektedir.

Çözünürlük düşüşünün her mimariyi eşit etkilemediği de görülmektedir: kayıp %9,4 ile %14,7
arasında değişmektedir. SegFormer-B0'ın en az kaybeden model olması, mimarinin konum
kodlaması kullanmamasıyla tutarlıdır [2].

> **Sonuç:** uç cihaz için model seçimi yayımlanmış doğruluğa bakılarak yapılamaz. Aday
> modeller dağıtım koşulunda ölçülmelidir.

## 5.2 DOĞRULUKTA AYIRT EDİLEBİLİR BİR KAZANAN YOKTUR

Çizelge 5.1'deki ölçülen değerler bir sıralama üretmektedir, ancak bu sıralamanın
farklarının gerçek olup olmadığı ayrıca test edilmelidir. Beş model **aynı 500 doğrulama
karesinde** puanlanmış, kare başına skorlar saklanmış ve her çift için fark eşleştirilmiş
bootstrap ile değerlendirilmiştir.

Çizelge 5.2. Eşleştirilmiş model karşılaştırması (500 ortak kare, %95 bootstrap).

| Karşılaştırma | Ortalama fark | %95 aralık | Sonuç |
|---|---|---|---|
| SegFormer-B0 − DDRNet-23-slim | +0,0000 | [−0,0050, +0,0049] | **ayırt edilemez** |
| PIDNet-M − BiSeNetV2 | +0,0050 | [−0,0001, +0,0108] | ayırt edilemez |
| SegFormer-B0 − PIDNet-M | +0,0063 | [+0,0012, +0,0120] | ayrışıyor |
| PIDNet-M − DDRNet-23-slim | −0,0063 | [−0,0112, −0,0012] | ayrışıyor |
| SegFormer-B0 − PIDNet-S | +0,0195 | [+0,0147, +0,0244] | ayrışıyor |
| PIDNet-S − DDRNet-23-slim | −0,0195 | [−0,0240, −0,0149] | ayrışıyor |

On model çiftinin sekizi ayrışmakta, **ilk ikisi ayrışmamaktadır**. Kare başına ortalama
mIoU her iki model için de 0,5468'dir.

**İki istatistik neden farklı söylüyor?** Veri kümesi mIoU'su 19 sınıfın ortalamasıdır ve
her sınıfa, kaç karede göründüğünden bağımsız olarak eşit ağırlık verir. Kare başına mIoU
ise her kareye eşit ağırlık verir. Sınıf kırılımı farkı açıklamaktadır: SegFormer-B0
19 sınıfın 14'ünü kazanmakta ve kazandıkları ağırlıklı olarak **ince yapılar** olmaktadır
(direk +0,0574, terrain +0,0422, trafik ışığı +0,0354, insan +0,0325); DDRNet-23-slim'in
kazandığı beş sınıf ise ağırlıklı olarak büyük araçlardır (otobüs −0,0577, tren −0,0448).

Şekil 5.2 sınıf bazlı IoU dağılımını göstermektedir.

**[Şekil 5.2: 02_per_class_iou.pdf]**

Şekil 5.2. Dağıtım çözünürlüğünde sınıf bazlı IoU.

> **Sonuç:** doğrulukta bir kazanan değil, **ayırt edilemez bir tepe grubu** vardır. Bu,
> model seçimini otomatik olarak diğer eksenlere — enerji ve dayanıklılığa — bırakmaktadır.

Bu rapor bu nedenle hiçbir yerde "en doğru model" ifadesini kullanmamaktadır.

## 5.3 NESNE DÜZEYİNDE KONUMLANDIRMA

Piksel metrikleri nesne bütünlüğüne duyarsızdır: büyük bir engelin üçte birini bulan model
ile aynı engeli üç ayrı parça olarak bulan model aynı puanı alır. Bu nedenle bileşen
düzeyinde ölçüm ayrıca yapılmıştır.

Çizelge 5.3. Bileşen düzeyinde konumlandırma (100 Cityscapes val karesi, 10 dikkat sınıfı).

| Mimari | Bileşen kapsama | En iyi bileşen IoU | Tahmin/GT bileşen oranı |
|---|---|---|---|
| SegFormer-B0 | **0,6780** | 0,4351 | **1,53×** |
| PIDNet-S | 0,6655 | 0,4456 | 1,00× |
| DDRNet-23-slim | 0,6428 | **0,4630** | 0,86× |

SegFormer-B0 gerçek nesnelerin daha büyük bir kısmını bulmakta, ancak onları 1,53 kat
fazla parçaya bölmektedir. DDRNet-23-slim'in davranışı terstir: daha az nesne bulmakta,
bulduğunu daha bütün bulmaktadır.

Bu yalnızca bir metrik farkı değildir. Zamansal izleyici ve risk sıralaması **bileşenler
üzerinde** çalışmaktadır; ikiye bölünmüş bir nesne, dikkat için yarışan iki iz demektir.
Yani yukarı akıştaki parçalanma, aşağı akışta doğrudan kararsızlığa dönüşmektedir.

Ölçülen bileşenler sınıf-bağlantılı bölgelerdir, **örnek (instance) değildir**; yan yana
duran aynı sınıftan iki araç burada tek bileşendir.

## 5.4 KALİBRASYON: HAVUZLANMIŞ HATA YANILTIYOR

Kalibrasyon, modelin ürettiği güven değerinin gerçek doğrulukla örtüşüp örtüşmediğidir
[10]. Şekil 5.3 güvenilirlik diyagramlarını göstermektedir.

**[Şekil 5.3: 03_reliability_diagrams.pdf]**

Şekil 5.3. Güvenilirlik diyagramları.

İlk bulgu, modelin **kesinlikle yanıldığı** piksellerdeki davranışıdır. Anomali pikselleri
19 sınıfın hiçbiri değildir; model ne tahmin ederse etsin hatalıdır. Şekil 5.7 bu
piksellerdeki güveni göstermektedir.

**[Şekil 5.7: 09_overconfidence.pdf]**

Şekil 5.7. Kesin yanlış piksellerde güven düzeyi.

Model, kesinlikle yanıldığı piksellerde bile %73–84 arası güven vermekte; güveni normal
piksellere göre yalnızca %4,4–10,5 düşmektedir. Bu, maksimum softmax olasılığının neden
zayıf bir open-set skoru olduğunu doğrudan açıklamaktadır.

### 5.4.1 Sınıf Bazlı Kalibrasyon Hatası

Bu çalışmada başlangıçta raporlanan ECE değerleri bütün pikselleri tek bir güven
histogramında toplamaktaydı. Bir sürüş sahnesinin **%37,65'i yol** sınıfıdır ve model bu
sınıfta hem çok emin hem de haklıdır; bu kütlenin küçük sınıflardaki aşırı güveni
maskeleyebileceği değerlendirilmiş ve ölçülmüştür.

Çizelge 5.4. Havuzlanmış ve sınıf bazlı ECE (200 kare, kare başına 20.000 piksel).

| Mimari | Havuzlanmış ECE | Sınıf bazlı ECE | Oran | En kötü sınıf |
|---|---|---|---|---|
| SegFormer-B0 | 0,0148 | **0,0446** | 3,01× | fence 0,1223 |
| PIDNet-S | 0,0323 | **0,0776** | 2,40× | pole **0,2598** |
| DDRNet-23-slim | 0,0409 | **0,0621** | 1,52× | pole 0,1744 |

Havuzlanmış ECE her modelde **1,5 ile 3 kat iyimserdir**. Kaynağı doğrudan görünmektedir:
`road` sınıfı piksellerin %37,65'ini kaplamakta ve her modelde en iyi kalibre sınıflardan
biri olmaktadır.

Kalibrasyon **sıralaması** da değişmektedir: havuzlanmış ECE'ye göre PIDNet-S,
DDRNet-23-slim'den daha iyi kalibre görünürken sınıf bazlı hesapta sıralama tersine
dönmektedir. Yani hangi ECE'nin raporlandığı, hangi modelin daha iyi kalibre olduğu
sorusunun cevabını değiştirmektedir.

En kötü kalibre sınıflar her modelde ince yapılardır. PIDNet-S direk sınıfında 0,2598 ECE
vermektedir; bu, kendi havuzlanmış değerinin sekiz katıdır ve aşırı-güven işareti
pozitiftir — yani model yanıldığı yerde emindir.

### 5.4.2 Rakip Açıklamanın Ölçülmesi: Nadirlik mi, İncelik mi?

Bu raporun birden fazla ölçümü aynı sınıflara işaret etmektedir. Sunulan mekanizma çıktı
stride'ıdır, ancak aynı kurbanları öngören ikinci bir mekanizma vardır: bu sınıflar aynı
zamanda **nadir** sınıflardır. Bir açıklamanın öne sürülüp diğerinin ölçülmeden
bırakılması kabul edilebilir olmadığından, sınıf dağılımı ölçülmüştür.

500 doğrulama karesindeki 917.018.489 etiketli piksel üzerinde sınıf dengesizliği
**473 kat**'tır (`road` %37,651, `motorcycle` %0,080). Piksel payı ile başarım arasındaki
ilişki:

- ρ(piksel payı, sınıf IoU) = **+0,68**
- ρ(piksel payı, sınıf ECE) = **−0,52**

Yani nadirlik gerçek bir etkendir. Ancak tek etken değildir ve bunu gösteren şey
sıralamanın kendisi değil, içindeki **ayrışma**dır:

Çizelge 5.5. Nadirlik ve kalibrasyon hatası ayrışması (ECE değerleri PIDNet-S).

| Sınıf | Piksel payı | Karelerde görülme | ECE | Şekil |
|---|---|---|---|---|
| motorcycle | %0,080 (en nadir) | %18,6 | **0,0288** | kompakt |
| truck | %0,301 | %16,0 | 0,0345 | kompakt |
| traffic light | %0,197 | %58,2 | **0,1499** | ince |
| pole | %1,479 | %98,2 | **0,2598** | ince |

`pole` sınıfı `motorcycle`'dan **18 kat daha sık** görülmekte ve karelerin neredeyse
tamamında bulunmaktadır; buna rağmen kalibrasyon hatası orada **9 kat** daha kötüdür.
Eşleşmiş nadirlikte de aynı durum geçerlidir: `truck` (%0,301) ile `traffic light`
(%0,197) benzer nadirlikteyken ECE değerleri 4,3 kat farklıdır.

> **Sonuç:** sınıf dengesizliği başarımı etkilemektedir, ancak bu çalışmadaki başarısızlık
> örüntüsünü açıklamamaktadır. Ayıran şey nadirlik değil **şekil**tir — ince ve uzun
> yapılar, aynı nadirlikteki kompakt sınıflardan sistematik olarak daha kötü
> segmentlenmekte ve daha kötü kalibre edilmektedir. Kaba çıktı ızgarası bunu öngörür;
> nadirlik öngörmez.

Bu gözlemsel bir ayrışmadır, kontrollü bir deney değildir. Kesin kanıt, aynı sınıfın farklı
stride değerlerinde ölçülmesi olurdu; bu, Bölüm 5.7'de tek mimari için yapılmıştır.

## 5.5 SÜRÜLEBİLİR ALAN

Sürülebilir alan belirleme başarımı, Cityscapes doğrulama kümesinin yol maskeleri
kullanılarak sayısal olarak ölçülmüştür.

Çizelge 5.6. Sürülebilir alan ölçümü (200 kare, ignore pikselleri dışlanmıştır).

| Mimari | Yol IoU | Sınır F1 (1 px) | Sınır F1 (8 px) | **Yanlış-sürülebilir** | Yol parçası |
|---|---|---|---|---|---|
| SegFormer-B0 | **0,9653** | **0,2336** | **0,6143** | 0,0091 | 8,62 |
| DDRNet-23-slim | 0,9622 | 0,1683 | 0,5817 | **0,0064** | **4,54** |
| PIDNet-M | 0,9617 | 0,1581 | 0,5651 | 0,0065 | 4,92 |
| PIDNet-S | 0,9614 | 0,1604 | 0,5651 | 0,0076 | 4,63 |
| BiSeNetV2 | 0,9602 | 0,1717 | 0,5910 | 0,0097 | 5,25 |

**Yanlış-sürülebilir oran**, aracın gireceği ama yol olmayan piksellerin oranıdır ve
güvenlik açısından anlamlı olan ölçüttür. Yol IoU değerlerinin tamamı 0,5 puanlık bir
aralığa sıkışmaktadır — yol kolay bir sınıftır — buna karşılık yanlış-sürülebilir oran
0,0064 ile 0,0097 arasında **1,5 kat** değişmektedir. Mimarileri ayıran ölçüt budur.

Sınır uyumu iki toleransta verilmiştir. 1 piksel toleransı, kaba logit ızgarasından
büyütülmüş bir maskeden çözünürlüğünün izin vermediği bir kesinlik istemektedir; tek başına
raporlanması modelin başarısızlığı gibi okunurdu, oysa bir çözünürlük sınırıdır. 8 piksel
toleransı bütün mimariler için **aynı** tutulmuştur, çünkü tolerans modele göre
esnetilseydi daha ince ızgarada çalışan modelin üstünlüğü metriğin içinde kaybolurdu.

### 5.5.1 Ego Koridoru Adımının Ölçülmüş Karşılığı

Ham yol maskesinden ego koridorunun ayrılması bir tasarım tercihiydi; ne kazandırdığı
ölçülmemişti. Beş mimaride de aynı yönde sonuç vermektedir:

Çizelge 5.7. Ego koridoru adımının etkisi.

| Mimari | Yanlış-sürülebilir: yol → koridor | Değişim | IoU bedeli |
|---|---|---|---|
| PIDNet-M | 0,0065 → 0,0048 | **−%26,2** | −0,53 puan |
| PIDNet-S | 0,0076 → 0,0058 | −%23,7 | −0,57 puan |
| SegFormer-B0 | 0,0091 → 0,0071 | −%22,0 | −0,53 puan |
| BiSeNetV2 | 0,0097 → 0,0077 | −%20,6 | −0,67 puan |
| DDRNet-23-slim | 0,0064 → 0,0052 | −%18,8 | −0,54 puan |

Koridor seçimi, ego konumuna bağlı olmayan yol bileşenlerini atmakta — yani tam olarak
sahte yol lekelerini elemektedir. Yaklaşık 0,55 puan IoU karşılığında yanlış-sürülebilir
piksellerin beşte biri ile dörtte biri arası elenmektedir. Güvenlik açısından bu takas
doğru yöndedir: kaybedilen şey doğru yolun bir kısmı, kazanılan şey yanlış yola
girmemektir.

## 5.6 OPEN-SET YOL TEHLİKESİ ALGILAMA

Open-set başarımı, hiçbir modelin eğitilmediği 60 karelik piksel etiketli gerçek yol
tehlikesi verisi üzerinde ölçülmüştür [6]. Şekil 5.5 skor karşılaştırmasını ve tehlike
türü kırılımını göstermektedir.

**[Şekil 5.5: 05_open_set.pdf]**

Şekil 5.5. Open-set skorları ve tehlike türüne göre kırılım.

Çizelge 5.8. Belirsizlik skorlarının anomali ayırt etme gücü (PIDNet-S).

| Skor | AUROC | AP | FPR95 |
|---|---|---|---|
| Energy | **0,6670** | **0,1562** | 0,8608 |
| Maximum logit | 0,6617 | 0,1539 | 0,8674 |
| Normalize entropi | 0,6022 | 0,1322 | 0,9399 |
| Maximum softmax olasılığı | 0,5693 | 0,1163 | 0,9462 |

`energy > max-logit > entropi > MSP` sıralaması, enerji tabanlı OOD literatürünün
bildirdiği sıralamayla birebir aynıdır [9]. Bu, uygulamanın doğruluğunu bu sayılardan
bağımsız olarak doğrulamaktadır.

Çizelge 5.9. Tehlike türüne göre open-set başarımı (energy skoru).

| Tehlike türü | AUROC | AP | Anomali piksel |
|---|---|---|---|
| Araç | 0,7070 | 0,1689 | 6.260 |
| Koni | 0,6884 | 0,0678 | 4.552 |
| Hayvan | 0,6717 | 0,2128 | 78.711 |
| Engel | 0,6276 | 0,1189 | 28.190 |
| **Kayıp yük** | **0,4805** | 0,0062 | 419 |

> **Güvenlik bulgusu:** model yola düşmüş yükte AUROC 0,4805 vermektedir — yani rastgele
> tahminden **daha kötüdür**. Sistem, yol üzerinde duran bir kargoyu ayırt edememektedir.

Bu bulgu literatürdeki karşılığıyla tutarlıdır: SegmentMeIfYouCan kıyaslaması yol
engellerini ayrı bir görev olarak tanımlamaktadır, çünkü genel anomali yöntemleri bu
görevde zayıf kalmaktadır [11].

## 5.7 UÇ CİHAZ MALİYETİ VE ÇIKTI STRIDE'I

Şekil 5.6, doğruluk-enerji ve open-set-gecikme ödünleşimlerini göstermektedir.

**[Şekil 5.6: 06_pareto.pdf]**

Şekil 5.6. Doğruluk ↔ enerji ve open-set ↔ gecikme ödünleşimleri.

Çizelge 5.10. Uç cihaz ölçümleri (Jetson Orin Nano Super, 25 W, 600 sn sürdürülen yük).

| Mimari | Motor | Kare | FPS | Ort. güç | J/kare | Tepe RAM | Logit çıktısı |
|---|---|---|---|---|---|---|---|
| PIDNet-M | 11,15 ms | 88,50 ms | 10,82 | 8,75 W | 0,808 J | 2.632 MiB | 64×128 |
| PIDNet-S | 5,07 ms | 80,77 ms | 11,87 | 7,89 W | 0,665 J | 2.601 MiB | 64×128 |
| **DDRNet-23-slim** | **3,81 ms** | **77,43 ms** | **12,36** | **7,79 W** | **0,630 J** | 2.541 MiB | 64×128 |
| SegFormer-B0 | 15,04 ms | **270,82 ms** | **3,57** | 7,93 W | **2,221 J** | 2.605 MiB | **128×256** |
| BiSeNetV2 | 13,45 ms | 92,36 ms | 10,34 | 8,36 W | 0,809 J | 2.530 MiB | 64×128 |

Hiçbir modelde termal kısıtlama gözlenmemiştir (GPU sıcaklığı 56–59 °C); güç tüketimi
25 W bütçesinin üçte biri düzeyindedir.

### 5.7.1 SegFormer Anomalisi

SegFormer-B0'ın TensorRT motoru DDRNet-23-slim'den yalnızca **11,23 ms** yavaştır, ancak
uçtan uca karesi **193,39 ms** yavaştır — motor farkının **17 katı**.

Nedeni çıktı çözünürlüğüdür: SegFormer-B0 logitlerini stride-4'te (128×256 = 32.768
piksel), diğer dört mimari stride-8'de (64×128 = 8.192 piksel) üretmektedir. CPU
tarafındaki her aşamaya **dört kat piksel** girmektedir.

> **Uç cihazda bir modelin sistem maliyetini belirleyen şey FLOP sayısı veya hızlandırıcı
> gecikmesi değil, segmentasyon başının çıktı stride'ıdır** — çünkü CPU tarafındaki algı
> yığını çıktı piksel sayısıyla ölçeklenir, hızlandırıcı ölçeklenmez.

Bu maliyetin giderilebilir bir entegrasyon artığı olup olmadığı ölçülmüştür. Logitleri
stride-8'e indirmek post-processing'i **3,98 kat** hızlandırmakta, ancak **2,03 mIoU'ya**
mal olmaktadır (0,6721 → 0,6518, 60 kare). Yani stride-4 çıktısı gerçek doğruluk
taşımaktadır.

> Uç cihaz için doğru soru "bu maliyet kaldırılabilir mi" değil, **"bu doğruluk bu enerjiye
> değer mi"**dir.

### 5.7.2 Çıktı Stride'ının Tekrar Eden İzi

Çıktı stride'ı bu raporda beş bağımsız ölçümde aynı sonuca işaret etmektedir:

Çizelge 5.11. Çıktı stride'ının farklı ölçümlerdeki izi.

| Ölçüm | Bulgu | Bölüm |
|---|---|---|
| Sınıf bazlı IoU | İnce yapılarda +0,032…+0,057 üstünlük | 5.2 |
| Sınır F1 (1 px) | %36–48 daha iyi | 5.5 |
| Sınıf bazlı ECE | En kötü sınıf her modelde direk/çit | 5.4 |
| Bileşen parçalanması | 1,53 kat daha parçalı | 5.3 |
| Post-processing maliyeti | 3,98 kat, 2,03 mIoU bedeliyle | 5.7 |

Beş ayrı ölçüm, tek bir mimari parametreye bağlanmaktadır.

## 5.8 OLUMSUZ KOŞULLARA DAYANIKLILIK

Şekil 5.4, dört gerçek olumsuz koşulda beş mimarinin başarımını göstermektedir.

**[Şekil 5.4: 04_acdc_conditions.pdf]**

Şekil 5.4. Gerçek olumsuz koşullarda doğruluk ve kalibrasyon.

Çizelge 5.12. ACDC koşullarında mIoU (406 kare).

| Mimari | Temiz | Sis | Kar | Yağmur | **Gece** | Gece kaybı | Gece ECE |
|---|---|---|---|---|---|---|---|
| SegFormer-B0 | 69,34 | 59,93 | 46,33 | 45,71 | **20,59** | −%70,3 | 0,3150 |
| PIDNet-S | 67,68 | 57,14 | 42,79 | 39,64 | 14,97 | −%77,9 | 0,2694 |
| BiSeNetV2 | 66,02 | 46,59 | 36,82 | 37,75 | 13,34 | −%79,8 | 0,2712 |
| PIDNet-M | 68,47 | 60,29 | 41,53 | 42,64 | 12,54 | −%81,7 | 0,3812 |
| DDRNet-23-slim | 68,50 | 56,99 | 38,75 | 44,59 | **7,12** | **−%89,6** | **0,4626** |

Üç bulgu öne çıkmaktadır.

**Birincisi, gece bütün mimarilerde yıkıcıdır.** En dayanıklı model bile doğruluğunun
%70'ini kaybetmektedir. Daha önemlisi, kalibrasyon her modelde **8,7 ile 14,0 kat**
bozulmaktadır (ECE ≈ 0,03'ten 0,27–0,46'ya). Model en çok yanıldığı koşulda yanıldığını
bilmemektedir; bu, sessiz başarısızlığın tanımıdır.

**İkincisi, mimariler eşit çökmemektedir.** SegFormer-B0 gece doğruluğunun %29,7'sini
korurken DDRNet-23-slim yalnızca %10,4'ünü korumakta ve 7,12 mIoU ile işlevsiz hâle
gelmektedir.

**Üçüncüsü, hız ve enerji kazananı dayanıklılık kaybedenidir.** DDRNet-23-slim en hızlı
(77,43 ms) ve en verimli (0,630 J/kare) modeldir; aynı zamanda olumsuz koşullara en
kırılgan olanıdır. Bu, model seçimine dördüncü bir eksen eklemektedir.

### 5.8.1 Sentetik Bozulma Gerçek Koşulun Yerini Tutmuyor

Şekil 5.8, sentetik bozulmalara belirsizlik tepkisini göstermektedir.

**[Şekil 5.8: 08_synthetic_shift.pdf]**

Şekil 5.8. Sentetik bozulmaya belirsizlik tepkisi.

Çizelge 5.13. Sentetik bozulma altında belirsizlik (15 kare, PIDNet-S).

| Koşul | Ortalama entropi | Temize oran | Düşük-güven piksel |
|---|---|---|---|
| Temiz | 0,1709 | 1,00× | %3,55 |
| Sis | 0,2009 | 1,18× | %6,76 |
| Kar | 0,1906 | 1,12× | %6,01 |
| Yağmur | 0,1736 | 1,02× | %4,33 |
| Gece | 0,1703 | **1,00×** | %3,95 |

Sentetik "gece" (parlaklık düşürme) belirsizlik sinyalinde **hiçbir tepki** yaratmamakta
(1,00×), buna karşılık gerçek ACDC gecesinde doğruluk %78 düşmektedir. Sentetik bozulma,
gerçek koşul kaymasının yerine geçememektedir; bu, sentetik stres testlerine dayanan
çalışmalar için doğrudan bir uyarıdır.

## 5.9 BAĞLAMSAL RİSK VE ZAMANSAL DAVRANIŞ

Şekil 5.9 ve Şekil 5.10 nitel sonuçları göstermektedir.

**[Şekil 5.9: qualitative/models_same_frame.png]**

Şekil 5.9. Aynı karede beş mimarinin çıktısı.

**[Şekil 5.10: qualitative/conditions_pidnet_s.png]**

Şekil 5.10. PIDNet-S çıktısı: temiz, sis ve gece koşulları.

Risk sıralaması yedi özellikli açıklanabilir bir füzyonla üretilmektedir. Ölçülemeyen
özellikler sıfır **değerle** değil sıfır **ağırlıkla** dışlanmakta, böylece ölçülmemiş bir
sinyalin "risk yok" gibi görünmesi ve bütün skorları aşağı çekmesi engellenmektedir.
Üretilen skor bir **operasyonel dikkat sıralamasıdır**, fiziksel risk olasılığı değildir;
kayıtlar bunu `calibrated_physical_risk_probability: false` alanıyla belirtmektedir.

Yedinci özellik olan zamansal kalıcılık, tek karede üretilemediği için sıfır ağırlıklıdır.
Dışlamanın bedeli 150 ardışık kare üzerinde ölçülmüştür:

Çizelge 5.14. Zamansal davranış (150 ardışık kare, PIDNet-S).

| Ölçüt | Değer |
|---|---|
| Bölge gözlemi | 5.208 |
| Adil değerlendirilen iz | 1.382 |
| **Tek karelik iz** | **692 — %50,1** |
| **Medyan iz ömrü** | **1 kare** |
| Birinci sıradaki bölgesi değişen kare | %20,0 |
| Titreyen iz (3+ kare yaşayıp kategori ≥2 kez değişen) | %18,9 |

> **Bulgu:** sistemin işaretlediğinin **yarısı tek kare yaşamaktadır**. Yani tek-kare
> operasyonel dikkat, istisna olarak değil **baskın davranış olarak** titremektedir.

Ayrı bir bozulma kipi de ölçülmüştür: üç kareden uzun yaşayan izlerin %18,9'u risk
kategorisini en az iki kez değiştirmektedir. Bu, kaybolan bir uyarıdan farklı ve
tartışmalı biçimde daha kötü bir hatadır — kendini sürekli yalanlayan kalıcı bir uyarı.

Bu ölçümün sınırı belirtilmelidir: demoVideo etiketsizdir. Ölçülen şey izlerin **geçici**
olduğudur, **yanlış** olduğu değil; bir kare görünüp kaybolan gerçek bir yaya ile bir
karelik segmentasyon gürültüsü bu veriyle ayırt edilememektedir.

## 5.10 FP16 DAĞITIM SADAKATİ

Bu rapordaki bütün doğruluk sayıları FP32 ONNX grafiğinden ölçülmüştür; cihaza dağıtılan
ise TensorRT FP16 motorudur. TensorRT belgeleri, indirgenmiş hassasiyetin doğruluk kaybına
yol açabileceğini belirtmektedir [13]. Dolayısıyla dağıtılan modelin doğruluğu ayrıca
ölçülmüştür.

Ölçüm, her iki hassasiyet **aynı koşuda aynı 100 kare** üzerinde puanlanarak yapılmıştır.

Çizelge 5.15. FP16 dağıtım sadakati (100 kare, eşleştirilmiş).

| Mimari | FP16 motor | Aynı koşuda FP32 | Eşleştirilmiş fark | %95 aralık | Piksel uyumu |
|---|---|---|---|---|---|
| PIDNet-S | 0,6670 | 0,6670 | +0,000019 | [−0,000925, +0,001016] | %99,9728 |
| DDRNet-23-slim | 0,6597 | 0,6597 | −0,000005 | [−0,000104, +0,000083] | %99,9712 |
| SegFormer-B0 | 0,6950 | 0,6951 | −0,0000004 | [−0,000076, +0,000076] | %99,9725 |

Üç mimaride de fark sıfırdan ayırt edilememektedir. Söylenebilecek şey **bu ölçümün
algılayabildiği bir doğruluk bedeli olmadığıdır**; "iki hassasiyet aynıdır" değil, çünkü
piksellerin %0,03'ü değişmekte, ancak değişenler metriği hareket ettirmemektedir.

### 5.10.1 Eşleştirmenin Önlediği Yanlış Sonuç

Aynı kayıtlar, eşleştirme yapılmasaydı elde edilecek sonucu da içermektedir. FP16 değerleri
100 kare üzerinde, referans FP32 değerleri ise ayrı bir koşuda 500 kare üzerinde
ölçülmüştür:

Çizelge 5.16. Eşleştirilmemiş karşılaştırmanın verdiği yanıltıcı sonuç.

| Mimari | FP16 (100 kare) | FP32 referans (500 kare) | **Naif fark** |
|---|---|---|---|
| PIDNet-S | 0,6670 | 0,6768 | −0,0098 |
| DDRNet-23-slim | 0,6597 | 0,6850 | **−0,0253** |
| SegFormer-B0 | 0,6950 | 0,6934 | +0,0016 |

Bu tablo okunduğunda "FP16 dağıtım DDRNet-23-slim'e 0,0253 mIoU'ya mal oluyor" sonucuna
varılırdı. Bu **yanlış** olurdu: aynı koşudaki FP32 ölçümü de 0,6597 vermektedir, yani
farkın tamamı kare alt kümesinden kaynaklanmaktadır.

> Ayrı ölçülmüş iki sayıyı yan yana koymak, ölçüldüğü sanılan şeyi ölçmez. Eşleştirilmiş
> biçim kullanılmasaydı bu rapor, var olmayan bir FP16 cezasını — üstelik en büyüğünü,
> enerji açısından en verimli modelin üzerine — yazacaktı.

Sıralamanın kare alt kümesine bağlılığı burada üçüncü kez gözlenmektedir: 100 karelik alt
kümede sıralama SegFormer-B0 > PIDNet-S > DDRNet-23-slim iken, 500 karede
SegFormer-B0 > DDRNet-23-slim > PIDNet-S'tir.

## 5.11 GERİ ÇEKİLEN İKİ SONUÇ

Çalışma sırasında iki sonuç yanlış çıkmış ve geri çekilmiştir. İkisi de burada
belirtilmektedir.

**Birincisi: "doğruluk arttıkça open-set güvenliği düşüyor" (ρ = −0,90).** Bu korelasyon
**yayımlanmış** mIoU değerleriyle hesaplanmıştı. Kendi dağıtım koşulumuzda ölçülen
değerlerle aynı hesap ρ = **+0,30** vermektedir; ilişki yoktur. Ayrıca
ρ(yayımlanmış, ölçülen) = **+0,10**'dur. Yani −0,90'ı üreten şey mimarilerin bir özelliği
değil, yayımlanmış sıralamanın dağıtım koşuluna taşınmamasıdır — bu ise zaten Bölüm 5.1'de
sunulan bulgudur.

**İkincisi: "SegFormer-B0'ın uçtan uca maliyeti giderilebilir bir entegrasyon artığıdır."**
Ölçüldüğünde maliyetin gerçek doğruluk taşıdığı görülmüştür (Bölüm 5.7).

## 5.12 SINIRLAR

- **n = 5 mimari.** Bu ölçekte hiçbir korelasyon kesin kanıt değildir; nitekim ρ = −0,90
  bulgusu tam bu nedenle yanlış okunmuştur. Bu raporun sayısal iddiaları korelasyondan
  değil, aynı karelerde yapılan eşleştirilmiş karşılaştırmalardan gelmektedir.
- **Referans checkpoint'ler farklı eğitim reçeteleriyle üretilmiştir**; bu bir kontrollü
  ablasyon değil, yayımlanmış model karşılaştırmasıdır.
- **Kendi eğitimimiz** yayımlanmış checkpoint'lerin yaklaşık %0,7'si kadar örnek
  görmüştür; sonuçları sınırlı bütçe altında çok-domainli genelleme denemesi olarak
  konumlandırılmıştır.
- **Gerçek zaman hedefi karşılanamamıştır** (en iyi 12,36 FPS, hedef ≥ 20).
- **Zamansal ölçüm etiketsiz veri üzerindedir**; geçicilik ölçülmüş, yanlışlık
  ölçülmemiştir.
- **Uç cihaz ölçümleri tek bir cihazın tek bir yapılandırmasıdır**; TensorRT'nin çekirdek
  seçimi donanım ve sürücü sürümüne bağlıdır.
- **Mühürlü nihai test verisi açılmamıştır**; bu kasıtlı bir kısıttır.
