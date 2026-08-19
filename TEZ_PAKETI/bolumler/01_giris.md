# 1. GİRİŞ

## 1.1 PROBLEM TANIMI

Otonom sürüş ve sürücü destek sistemlerinin algı katmanı, sahneyi anlamlı sınıflara ayıran
semantik segmentasyon modelleri üzerine kuruludur. Bu modeller kapalı küme varsayımıyla
eğitilir: dünyanın, eğitim sırasında tanımlanmış sınıf kümesinden oluştuğu varsayılır.
Cityscapes veri setinde bu küme 19 sınıftan ibarettir [7].

Varsayım gerçek yolda bozulur. Yola düşmüş bir kasa, devrilmiş bir trafik konisi ya da
şeride girmiş bir hayvan, tanımlı sınıfların hiçbirine ait değildir. Böyle bir nesneyle
karşılaşan model bir hata mesajı üretmez; onu **en yakın gördüğü sınıfa atar** ve bu
atamadan yüksek güven bildirir.

Bu davranış tekil bir yanlış sınıflandırmadan daha ciddidir. Sistemin çıktısına dayanan bir
karar mekanizması, düşük güven bildirilmediği sürece o çıktının güvenilir olduğunu
varsayacaktır. Yani sorun modelin **yanılması** değil, **yanıldığını bildirememesidir**.

Bu durum, ISO 21448 (SOTIF) standardının konusu olan tehlike sınıfına girer: donanım
arızalanmadan, yazılım çökmeden, yalnızca amaçlanan işlevin yetersiz kalması nedeniyle
ortaya çıkan tehlikeler [17].

## 1.2 KAYNAK KISITININ GETİRDİĞİ İKİNCİ PROBLEM

Belirsizlik kestirimi ve open-set algılama, hesaplama maliyeti olan işlemlerdir. Bu
maliyet, bulut sunucularında ihmal edilebilirken uç cihazlarda belirleyicidir. Araç içi bir
sistem bulut bağlantısına güvenemez; işlemin cihaz üzerinde, sınırlı bir güç bütçesi içinde
ve gerçek zamana yakın hızda tamamlanması gerekir.

Bu iki gereksinim — güvenilir belirsizlik bildirimi ve sıkı kaynak bütçesi — birbirini
kısıtlamaktadır. Bu çalışmanın konusu bu kısıtın nicel olarak ölçülmesidir.

## 1.3 LİTERATÜR TARAMASI

Literatür taraması sistematik bir akışla yürütülmüştür. Kaynak havuzu, dört ana başlıkta
yapılandırılmış aramalarla oluşturulmuş 36 tarama dokümanından derlenmiştir: gerçek zamanlı
semantik segmentasyon, belirsizlik kestirimi ve kalibrasyon, open-set segmentasyon, uç
cihaz dağıtımı.

Şekil 1.1, tarama ve eleme akışını göstermektedir.

**[Şekil 1.1: D5_prisma_flow.pdf]**

Şekil 1.1. Literatür tarama akışı (PRISMA).

Akışın sayıları elle yazılmamış, korpustan türetilmiştir; aşamaların toplamı kayıt
yazılmadan önce doğrulanmaktadır. Tanımlama aşamasında 36 dokümandan 1.480 benzersiz kayıt
elde edilmiştir. Tarama aşamasında 331 araç/dokümantasyon bağlantısı, 119 akademik olmayan
kaynak ve 459 sınıflandırılamayan kayıt çıkarılmış, uygunluk aşamasına 571 akademik yayın
kalmıştır. Bunlardan **19'u** dahil edilmiştir.

571'den 19'a düşüş bu çalışmanın bir sınırıdır ve öyle belirtilmelidir: literatür taraması
sistematik biçimde **tanımlanmış**, ancak kapsamlı biçimde **okunmamıştır**. Dahil edilen
19 kaynağın tamamı birincil kaynağından açılıp künyesi doğrulanmıştır; doğrulanmamış hiçbir
kayıt kaynakçaya girmemiştir.

### 1.3.1 Gerçek Zamanlı Semantik Segmentasyon

Gerçek zamanlı segmentasyon literatürü, doğruluk ile hesaplama maliyeti arasındaki
ödünleşimi farklı mimari stratejilerle ele almaktadır. İki dallı yapılar, yüksek
çözünürlüklü detay dalını düşük çözünürlüklü bağlam dalıyla birleştirir; DDRNet bu
yaklaşımı Deep Aggregation Pyramid Pooling Module ile genişletir [3]. BiSeNetV2, benzer
ayrımı Detail ve Semantic dalları ile kurar ve Guided Aggregation Layer ile birleştirir
[4]. PIDNet, bu iki dallı yapıya sınır bilgisini taşıyan üçüncü bir dal ekleyerek PID
denetleyici benzeşimi kurar [1]. SegFormer ise farklı bir yol izleyerek hiyerarşik bir
transformer kodlayıcıyı hafif bir MLP kod çözücüyle birleştirir ve konum kodlaması
kullanmaz; makale bunun çözünürlük değişimlerine dayanıklılık sağladığını bildirmektedir
[2].

Bu literatürün ortak bir zayıflığı, karşılaştırmaların çoğunlukla yalnızca hızlandırıcı
gecikmesini raporlamasıdır. Bu çalışmanın Bölüm 5.7'deki bulgusu, uçtan uca ölçüldüğünde
tablonun değiştiğini göstermektedir.

### 1.3.2 Belirsizlik Kestirimi ve Kalibrasyon

Modern derin ağların kötü kalibre olduğu, yani ürettikleri güven değerinin gerçek
doğruluktan yüksek olduğu bilinmektedir; Guo ve arkadaşları bu olguyu ve düzeltme yöntemi
olarak sıcaklık ölçeklemeyi tanımlamıştır [10]. Kalibrasyon hatasının standart ölçüsü
Beklenen Kalibrasyon Hatası'dır (ECE).

Bu çalışmanın Bölüm 5.4'teki katkısı, tek bir havuzlanmış ECE değerinin sürüş
sahnelerinde yanıltıcı olduğunu ölçmesidir.

### 1.3.3 Open-Set Algılama

Eğitim dağılımı dışındaki girdilerin tespiti için önerilen skorlar arasında maksimum
softmax olasılığı, entropi, maksimum logit ve enerji bulunmaktadır. Liu ve arkadaşları
enerji tabanlı skorun softmax tabanlı skorlardan daha iyi ayırt ettiğini savunmaktadır
[9]. Bu çalışmada dört skor da ölçülmüş ve literatürdeki sıralama bağımsız olarak
doğrulanmıştır (Bölüm 5.6).

Değerlendirme verisi bakımından RoadAnomaly, yol üzerindeki beklenmedik nesneleri piksel
düzeyinde etiketleyen bir kaynaktır [6]. SegmentMeIfYouCan kıyaslaması, yol engeli
segmentasyonunu ayrı bir görev olarak tanımlamaktadır; gerekçesi genel anomali
yöntemlerinin bu görevde zayıf kalmasıdır [11]. Bu çalışmanın Bölüm 5.6'daki kayıp yük
bulgusu bu gözlemle örtüşmektedir.

### 1.3.4 Uç Cihaz Dağıtımı

Uç cihaz dağıtımı literatürü ağırlıklı olarak model sıkıştırma, kuantalama ve donanıma
özgü derleme üzerinedir. Jetson platformunun termal davranışı ve güç tüketimi ayrı bir
inceleme alanıdır; Krišlaurks ve arkadaşları ortam sıcaklığının etkisini
karakterize etmiştir [14]. Bu çalışma, farklı bir Jetson modeli üzerinde çalışmakta
olduğundan o çalışmanın sayısal sonuçlarını değil, kısıtlamanın sıcaklık serisinden
okunması yöntemini kullanmaktadır.

### 1.3.5 Literatürdeki Boşluk

Yukarıdaki dört alan büyük ölçüde birbirinden bağımsız gelişmiştir. Gerçek zamanlı
segmentasyon çalışmaları doğruluk ve gecikmeyi, kalibrasyon çalışmaları güven kalitesini,
open-set çalışmaları anomali tespitini, uç cihaz çalışmaları ise donanım verimliliğini
raporlamaktadır.

**Bu eksenlerin aynı modeller üzerinde, aynı protokolle ve gerçek dağıtım koşulunda
birlikte ölçüldüğü çalışma sayısı azdır.** Bu çalışmanın konumu budur.

## 1.4 ÇALIŞMANIN AMACI

Bu çalışmanın amacı, kaynak kısıtlı bir uç cihazda çalışan, belirsizliğinin farkında ve
open-set girdilere dayanıklı bir yol tehlikesi algılama hattı tasarlamak ve bu hattı dört
eksende ölçmektir:

1. **Doğruluk** — dağıtım çözünürlüğünde semantik segmentasyon başarımı,
2. **Belirsizlik** — güven değerlerinin kalibrasyonu ve güvenilmez bölgelerin işaretlenmesi,
3. **Open-set** — eğitim dağılımında bulunmayan yol tehlikelerinin ayırt edilmesi,
4. **Uç maliyet** — gecikme, güç ve kare başına enerji.

Ölçümler NVIDIA Jetson Orin Nano Super üzerinde, 25 W güç modunda ve gerçek dağıtım
koşullarında yapılmıştır.

## 1.5 KATKILAR

Bu çalışmanın katkıları aşağıdadır. Her katkının dayandığı ölçüm ilgili bölümde
verilmiştir.

**1. Beş gerçek zamanlı mimarinin dört eksende ortak protokolle karşılaştırılması.**
Doğruluk, kalibrasyon, open-set başarımı ve uç cihaz maliyeti aynı modeller üzerinde, aynı
verilerle ve aynı donanımda ölçülmüştür (Bölüm 5).

**2. Yayımlanmış doğruluk sıralamasının dağıtım sıralamasını öngörmediğinin
gösterilmesi.** Yayımlanmış ve ölçülen sıralamalar arasındaki Spearman korelasyonu
ρ = +0,10'dur; yayımlanmış sıralamada sonuncu olan model dağıtım koşulunda öne
geçmektedir (Bölüm 5.1).

**3. Doğruluk farklarının istatistiksel olarak değerlendirilmesi.** Aynı karelerde
eşleştirilmiş bootstrap ile yapılan karşılaştırma, ilk iki modelin ayırt edilemediğini
göstermektedir (Bölüm 5.2). Bu, model seçimini enerji ve dayanıklılık eksenlerine
bırakmaktadır.

**4. Uç cihazda sistem maliyetinin segmentasyon başının çıktı stride'ı tarafından
belirlendiğinin ölçümle gösterilmesi.** Bir mimarinin hızlandırıcı gecikmesi 11,23 ms
fazlayken uçtan uca karesi 193,39 ms fazladır; nedeni CPU tarafına giren piksel sayısıdır.
Aynı parametre beş ayrı ölçümde tutarlı iz bırakmaktadır (Bölüm 5.7).

**5. Modele dokunmadan 1,88 kat hızlanma.** Kare bütçesinin ölçülmesi, maliyetin
hızlandırıcıda değil CPU tarafında olduğunu göstermiş; yapılan iki optimizasyon kare
süresini 151,67 ms'den 80,77 ms'ye indirmiştir. Çıktılar birebir aynıdır (Bölüm 4).

**6. Gerçek olumsuz koşullarda kalibrasyonun çöktüğünün gösterilmesi.** Gece koşulunda
kalibrasyon hatası 8,7 ile 14,0 kat artmaktadır; model en çok yanıldığı koşulda yanıldığını
bilmemektedir (Bölüm 5.8).

**7. Havuzlanmış kalibrasyon hatasının yanıltıcı olduğunun ölçülmesi.** Sınıf bazlı
hesaplandığında hata 1,5 ile 3 kat büyümekte ve kalibrasyon sıralaması değişmektedir
(Bölüm 5.4).

## 1.6 RAPORUN YAPISI

Bölüm 2, kullanılan veri setlerini, mimarileri, sistem mimarisini ve ölçüm protokolünü
tanımlar; bölümün sonunda bütün doğruluk sayılarının dayandığı veri bütünlüğü varsayımının
denetimi verilmiştir. Bölüm 3, uygulama sırasında karşılaşılan sorunları, kök nedenlerini
ve çözümlerini anlatır. Bölüm 4, kare bütçesinin ölçülmesini ve buna dayanan optimizasyonu
sunar. Bölüm 5, dört eksendeki bulguları ve tartışmayı içerir. Bölüm 6 sonuçları,
standartlarla ilişkiyi, girişimcilik ve sürdürülebilirlik boyutlarını ve gelecek çalışma
önerilerini verir.
