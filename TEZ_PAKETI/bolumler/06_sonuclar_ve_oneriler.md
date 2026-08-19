# 6. SONUÇLAR VE ÖNERİLER

## 6.1 SONUÇLAR

Bu çalışmada, kaynak kısıtlı bir uç cihazda çalışan belirsizlik farkındalıklı bir yol
tehlikesi algılama hattı tasarlanmış ve dört eksende ölçülmüştür. Elde edilen sonuçlar
aşağıda özetlenmiştir.

**Uç cihaz için model seçimi yayımlanmış doğruluğa bakılarak yapılamaz.** Yayımlanmış ve
dağıtım çözünürlüğünde ölçülen sıralamalar arasındaki korelasyon ρ = +0,10'dur; yayımlanmış
sıralamada sonuncu olan model ölçümde öne geçmektedir. Aday modeller, dağıtılacakları
koşulda ölçülmelidir.

**Doğruluk ekseninde ayırt edilebilir bir kazanan yoktur.** Aynı 500 karede yapılan
eşleştirilmiş karşılaştırma, ilk iki modeli ayırt edememektedir (+0,0000, %95 aralık
[−0,0050, +0,0049]). Bu nedenle model seçimi, ayrışan diğer eksenlere — enerji ve
dayanıklılığa — bırakılmalıdır. Bu iki eksenin kazananları ise farklıdır.

**Uç cihazda sistem maliyetini belirleyen şey model boyutu veya FLOP sayısı değil,
segmentasyon başının çıktı stride'ıdır.** Bir mimarinin hızlandırıcı gecikmesi 11,23 ms
fazlayken uçtan uca kare süresi 193,39 ms fazladır. Neden, CPU tarafındaki algı yığınının
çıktı piksel sayısıyla ölçeklenmesi, hızlandırıcının ise ölçeklenmemesidir. Aynı parametre
sınıf bazlı doğrulukta, sınır uyumunda, kalibrasyonda, bileşen parçalanmasında ve
post-processing maliyetinde tutarlı iz bırakmaktadır.

**Bu maliyet giderilebilir bir entegrasyon artığı değildir.** Çıktı çözünürlüğünü düşürmek
post-processing'i 3,98 kat hızlandırmakta, ancak 2,03 mIoU'ya mal olmaktadır. Uç cihaz için
doğru soru "bu maliyet kaldırılabilir mi" değil, "bu doğruluk bu enerjiye değer mi"dir.

**Gece koşulu ayrı ele alınmalıdır.** Bütün mimariler gece doğruluğunun en az %70'ini
kaybetmekte, kalibrasyon hatası 8,7 ile 14,0 kat artmaktadır. Model en çok yanıldığı
koşulda yanıldığını bilmemektedir. Bu, çalışma alanının (ODD) gündüzle sınırlanması ya da
belirsizlik sinyalinin aydınlatmaya duyarlı hâle getirilmesi gerektiğini göstermektedir.

**Sistem yol üzerindeki kayıp yükü ayırt edememektedir** (AUROC 0,4805, rastgeleden kötü).
Bu, çalışmanın çözmediği ancak nicel olarak ortaya koyduğu en önemli güvenlik boşluğudur.

**Gerçek zaman hedefi karşılanamamıştır** (en iyi 12,36 FPS, hedef ≥ 20). Ancak darboğazın
nerede olduğu ölçülmüştür: hızlandırıcının payı %5,4, geri kalanı CPU tarafıdır.

## 6.2 STANDARTLARLA İLİŞKİ

Bu bölümde, çalışmanın ilgili standartların hangi maddelerine karşılık geldiği ve hangi
maddelerine **karşılık gelmediği** ayrı ayrı belirtilmektedir. Bir standardın adını anmak,
o standarda uygunluk iddiası değildir.

### 6.2.1 ISO/IEC 25010 — Yazılım Ürün Kalite Modeli

ISO/IEC 25010, yazılım kalitesini sekiz karakteristiğe ayırmaktadır [15]. Bu çalışmada
dördü doğrudan ölçülmüş, dördü ölçülmemiştir.

Çizelge 6.1. ISO/IEC 25010 karakteristiklerinin bu çalışmadaki karşılığı.

| Karakteristik | Bu çalışmadaki karşılığı | Durum |
|---|---|---|
| Performans verimliliği — zaman davranışı | Gecikme p50/p95/p99, sürdürülen FPS, 600 sn yük | Ölçüldü |
| Performans verimliliği — kaynak kullanımı | Tepe RAM (2.530–2.632 MiB), güç, joule/kare | Ölçüldü |
| Güvenilirlik — olgunluk | 600 sn sürdürülen yükte termal kısıtlama yok (56–59 °C) | Ölçüldü |
| Taşınabilirlik — uyarlanabilirlik | PyTorch → ONNX → TensorRT FP16, sadakat doğrulamalı | Ölçüldü |
| İşlevsel uygunluk — doğruluk | mIoU, sınıf bazlı IoU, bileşen kapsama | Ölçüldü |
| Kullanılabilirlik | — | Ölçülmedi |
| Güvenlik (security) | — | Kapsam dışı |
| Bakım yapılabilirlik | 790 birim testi, her hata için regresyon testi | Kısmi |

ISO/IEC 25010 bir kalite **modelidir**, uygunluk sertifikası değildir. Bu çalışma
"25010 uyumludur" iddiasında bulunmamakta; standardın tanımladığı karakteristiklerden
dördünü ölçtüğünü, dördünü ölçmediğini belirtmektedir.

### 6.2.2 ISO/IEC/IEEE 29119 — Yazılım Testi

Test dokümantasyonu için sıkça anılan IEEE 829-2008 standardının yerini ISO/IEC/IEEE
29119-3 almıştır [16]. Bu çalışmadaki karşılıkları:

- **Test tasarımı.** Karşılaşılan her gerçek hata için bir regresyon testi yazılmıştır.
  Testler hatanın kendisini değil, hatanın **koşulunu** sabitler.
- **Test dokümantasyonu.** Her ölçüm kaydı `record_type`, `schema_version`,
  `model_sha256`, rastgelelik tohumu ve `scientific_measurement` alanlarını taşır; bir
  sonucun hangi kodla ve hangi ağırlıkla üretildiği kayıttan doğrulanabilir.
- **Karşılanmayan.** Biçimsel bir test planı ve test tamamlama raporu üretilmemiştir.

### 6.2.3 ISO 21448 (SOTIF) — Amaçlanan İşlevin Güvenliği

ISO 21448, sistemin arızalanmadığı ancak yetersiz kaldığı durumlardan doğan tehlikeleri ele
alır [17]. Bu çalışmanın bulgularının önemli bir kısmı bu sınıfa girmektedir:

- **Bilinmeyen tehlikeli senaryo:** eğitim dağılımında bulunmayan yol engelleri. Kayıp yük
  AUROC 0,4805 — sistem arızalı değildir, yalnızca fark edememektedir.
- **Çalışma alanı sınırı:** gece koşulunda doğruluğun %70–90 kaybı ve kalibrasyonun 8,7–14
  kat bozulması, çalışma alanının gündüzle sınırlanması gerektiğinin ölçülmüş
  gerekçesidir.
- **Aşırı güven:** model kesinlikle yanıldığı piksellerde %73–84 güven vermektedir. SOTIF
  açısından tehlikeli olan yanılmak değil, yanıldığını bilmemektir.

**Bu çalışma SOTIF sürecini uygulamamıştır.** Tehlike analizi, senaryo kataloğu, kabul
kriteri türetimi ve doğrulama kampanyası yapılmamıştır. Standart burada, ölçülen olguları
adlandıran kavramsal bir çerçeve olarak kullanılmaktadır; hiçbir uygunluk iddiası yoktur.

### 6.2.4 ISO 26262 — İşlevsel Güvenlik

ISO 26262 donanım ve yazılım arızalarını ele alır ve ASIL seviyesi, geliştirme süreci,
araç entegrasyonu ile doğrulama kampanyası gerektirir [18]. Bu çalışma bir araştırma
prototipidir; hiçbir ASIL seviyesi hedeflenmemiş ve hiçbir güvenlik gereksinimi
türetilmemiştir. Standart, **kapsam dışı olduğunu belirtmek için** anılmaktadır.

### 6.2.5 ISO 690 — Kaynak Gösterimi

Kaynakça ISO 690'a uygun biçimde, numaralı sistemle verilmiştir [19]. Uygulanan kural
şudur: listeye yalnızca birincil kaynağından açılıp künyesi doğrulanmış kayıt girmiştir.

## 6.3 GİRİŞİMCİLİK VE YENİLİKÇİLİK

### 6.3.1 Çözülen Pratik Problem

Uç cihaza model seçen bir ekibin bugün elindeki bilgi, model zoo'ların yayımladığı doğruluk
sıralamasıdır. Bu çalışma o sıralamanın dağıtım koşuluna taşınmadığını ölçmüştür
(ρ = +0,10). Pratik karşılığı doğrudandır: yanlış model seçen bir ekip ya gereksiz donanım
maliyetine katlanır ya da sahada yetersiz kalan bir sistem dağıtır.

### 6.3.2 Aktarılabilir Yöntem

Bu çalışmanın ürünü bir model değil, bir **seçim yöntemidir**:

1. Adayları dağıtım çözünürlüğünde ölç; yayımlanmış sayıya güvenme.
2. Farkların gerçek olup olmadığını eşleştirilmiş karşılaştırmayla test et.
3. Doğruluk ayrışmıyorsa seçimi enerji ve dayanıklılığa bırak.
4. Sistem maliyetini FLOP'tan değil çıktı stride'ından tahmin et.

Dört adımın hiçbiri bu veri kümesine veya bu donanıma özgü değildir.

### 6.3.3 Yenilikçi Bulgu

Uç cihazda sistem maliyetini segmentasyon başının çıktı stride'ının belirlediği bulgusu,
literatürde yaygın olarak vurgulanmamaktadır. Bunun nedeni, karşılaştırmaların çoğunun
yalnızca hızlandırıcı gecikmesini raporlamasıdır; uçtan uca ölçen bir çalışma bu farkı
görmek zorundadır.

### 6.3.4 Ürünleşme Yolu ve Engel

Prototipten ürüne giden yolda üç somut adım bulunmaktadır: gece için ayrı bir çalışma kipi,
piksel düzeyinde etiketli open-set verisi ve CPU tarafı maliyetinin daha da azaltılması.

Engel de açıkça belirtilmelidir: gerçek zaman hedefi hiçbir yapılandırmada
karşılanamamıştır. Sistem bugünkü hâliyle bir ürün değildir. Darboğazın model değil CPU
tarafı olduğu ölçüldüğü için yol açıktır, ancak henüz yürünmemiştir.

## 6.4 SÜRDÜRÜLEBİLİRLİK

Bu bölüm doğrudan ölçülmüş enerji değerlerine dayanmaktadır. Enerji, Jetson modülünün
dahili sensörlerinden 600 saniyelik sürdürülen yük boyunca kaydedilmiş; kare başına enerji
ortalama güç ve sürdürülen FPS'ten türetilmiştir.

### 6.4.1 Mimari Seçiminin Enerji Karşılığı

Çizelge 6.2. Kare başına enerji tüketimi.

| Mimari | J/kare | DDRNet-23-slim'e göre |
|---|---|---|
| DDRNet-23-slim | 0,630 | — |
| PIDNet-S | 0,665 | +%5,6 |
| PIDNet-M | 0,808 | +%28,3 |
| BiSeNetV2 | 0,809 | +%28,4 |
| SegFormer-B0 | 2,221 | +%252 |

DDRNet-23-slim'i PIDNet-M yerine seçmek kare başına **%22 enerji tasarrufu** sağlamaktadır.
Bölüm 5.2'de gösterildiği üzere bu seçim ölçülebilir bir doğruluk kaybı getirmemektedir.

> Doğruluk ekseninde modeller ayrışmadığında, enerji bedava bir tercih değil; **tek anlamlı
> tercihtir.**

### 6.4.2 Ölçek Etkisi

Tek bir cihaz kesintisiz çalıştığında 7,79 W ile yılda yaklaşık 68 kWh tüketmektedir. Aynı
kare sayısını en yüksek maliyetli mimariyle işlemek yaklaşık 241 kWh gerektirir. Bin
cihazlık bir filoda aradaki fark yılda 170 MWh'ı aşar.

Bu bir projeksiyondur, ölçüm değildir: tek cihazın ölçülmüş kare başına enerji değeri
çarpılmıştır. Soğutma, boşta kalma ve iletişim maliyetleri hesaba katılmamıştır.

### 6.4.3 Geliştirme Aşamasının Maliyeti

Sürdürülebilirlik yalnızca dağıtımın değil, geliştirmenin de meselesidir. Bu çalışmada
bilinçli olarak yapılan üç şey vardır:

- **Referans checkpoint'ler yeniden eğitilmemiştir.** Beş mimarinin 120.000–160.000 adımlık
  eğitimi tekrarlanmamış, yayımlanmış Apache-2.0 ağırlıklar kullanılmıştır.
- **Kendi eğitimimiz 2.500 adımda sınırlandırılmıştır.** Bu bir eksiklik olarak da
  belirtilmiştir; ancak tekrar tekrar eğitmek yerine ölçüm derinliğine yatırım yapmak
  bilinçli bir karardır.
- **Optimizasyon modele değil koda yapılmıştır.** Kare süresi 151,67 ms'den 80,77 ms'ye
  inmiş, motor değişmemiş ve çıktılar birebir aynı kalmıştır. Yeniden eğitim gerektirmeyen
  hızlanma, en düşük maliyetli hızlanmadır.

### 6.4.4 Ölçülmeyenler

Eğitim aşamasının toplam enerjisi ölçülmemiştir. Cihazın üretim ve imha ayak izi kapsam
dışıdır. Bu bölüm **kullanım aşaması** enerjisiyle sınırlıdır.

## 6.5 GELECEK ÇALIŞMALAR

**Kayıp yük tespiti.** Bu çalışmanın ortaya koyduğu en önemli boşluktur. Genel amaçlı
anomali skorları bu görevde yetersiz kalmaktadır; yol yüzeyine özgü bir yaklaşım
gerekmektedir.

**Gece için ayrı çalışma kipi.** Kalibrasyonun gece 8,7–14 kat bozulması, tek bir modelin
bütün koşulları karşılamasının mümkün olmadığını göstermektedir.

**Zamansal kalıcılığın dağıtımda etkinleştirilmesi.** İzlerin %50,1'inin tek kare yaşadığı
ölçülmüştür; zamansal filtreleme bu baskın bozulma kipine doğrudan karşılık gelmektedir.

**Çıktı stride'ının kontrollü incelenmesi.** Bu çalışmada stride etkisi gözlemsel olarak
beş ölçümde tutarlı bulunmuş, tek mimaride kontrollü olarak ölçülmüştür. Aynı mimarinin
farklı stride değerlerinde eğitilmesi, gözlemi kontrollü bir deneye dönüştürecektir.

**CPU tarafı maliyetinin azaltılması.** Gerçek zaman hedefine giden yol buradan
geçmektedir; kare bütçesinin %94,6'sı bu tarafta harcanmaktadır.
