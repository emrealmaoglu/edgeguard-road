# 4. OPTİMİZASYON

Bu bölüm, uç cihazda kare süresinin nereye harcandığının nasıl ölçüldüğünü, buna dayanarak
yapılan iki optimizasyonu ve elde edilen kazancı anlatmaktadır. Bölümün ana savı şudur:
gerçek zamanlılık, beklenenin aksine hızlandırıcıda değil **CPU tarafındaki algı yığınında**
kazanılıp kaybedilmektedir.

## 4.1 ÖLÇMEDEN OPTİMİZE ETMEME İLKESİ

Optimizasyon çalışması "ölç → ata → düzelt → yeniden ölç" döngüsü olarak yürütülmüştür.
Buradaki kritik adım **atama**dır: bir aşamanın yavaş olduğunu bilmek yetmez, kare
bütçesinin ne kadarını tükettiği bilinmelidir.

Bu nedenle önce, dağıtım hattının her aşaması gerçek cihazda ayrı ayrı zamanlanmıştır.
Şekil 4.1 ölçülen kare bütçesinin dağılımını göstermektedir.

**[Şekil 4.1: D4_frame_budget_flow.pdf]**

Şekil 4.1. Kare bütçesinin aşamalara dağılımı ve optimizasyonun etkisi.

## 4.2 KARE BÜTÇESİNİN ÖLÇÜLMESİ

Çizelge 4.1, PIDNet-S ile 60 kare üzerinde gerçek cihazda alınan aşama profilini
göstermektedir.

Çizelge 4.1. Kare bütçesi aşama profili (PIDNet-S, 60 kare, Jetson Orin Nano Super, 25 W).

| Aşama | İlk ölçüm | Optimizasyon sonrası | Pay |
|---|---|---|---|
| `derive_perception` | 96,05 ms | 43,64 ms | %46,4 |
| Ön işleme | 24,80 ms | 24,71 ms | %26,3 |
| Görüntü çözme | 13,97 ms | 14,00 ms | %14,9 |
| Güven ve entropi | 5,78 ms | 5,80 ms | %6,2 |
| **TensorRT motoru** | **5,09 ms** | **5,09 ms** | **%5,4** |
| Argmax | 0,75 ms | 0,74 ms | %0,8 |
| **Toplam** | **146,44 ms** | **93,98 ms** | |

> **Bulgu:** TensorRT motoru kare bütçesinin yalnızca **%5,4'ünü** tüketmektedir. Model
> mimarisini hızlandırmaya veya kuantalamaya yönelik her çaba, bu %5,4'lük dilim üzerinde
> çalışacaktı.

Bu sonuç optimizasyon stratejisini doğrudan belirlemiştir: çalışma modele değil,
bütçenin %94,6'sını tüketen CPU tarafına yöneltilmiştir.

## 4.3 BİRİNCİ OPTİMİZASYON: MESAFE DÖNÜŞÜMÜNÜN AYRILABİLİR HÂLE GETİRİLMESİ

**Durum.** Sürülebilir koridora uzaklık hesabı, piksel piksel ilerleyen bir
genişlik-öncelikli arama (BFS) kuyruğuyla yapılıyordu ve tek başına 9,20 ms tutuyordu.

**Gözlem.** Engel içermeyen bir 4-komşuluk ızgarasında genişlik-öncelikli arama, tam
olarak **L1 (Manhattan) mesafe dönüşümüne** eşittir. L1 mesafesi ise eksenlere göre
**ayrılabilir**: önce satırlar, sonra sütunlar boyunca kümülatif minimum alınarak aynı
sonuç elde edilir.

**Uygulama.** Python döngüsü, dört adet `numpy.minimum.accumulate` taramasıyla
değiştirildi.

**Sonuç.** 9,20 ms → 0,27 ms, yani **34 kat** hızlanma. Çıktı birebir aynıdır; eski
uygulama test içinde saklanarak iki uygulamanın aynı sonucu ürettiği rastgele maskeler
üzerinde doğrulanmıştır.

## 4.4 İKİNCİ OPTİMİZASYON: BİLEŞEN ETİKETLEMENİN VEKTÖRLEŞTİRİLMESİ

**Durum.** Bağlantılı bileşen etiketleme de piksel piksel Python araması yapıyordu ve
`derive_perception` bu işlemi kare başına on bir kez çağırıyordu.

**İlk deneme ve geri alınması.** Etiketlerin bütün-dizi maksimumlarıyla yayılmasına dayalı
vektörleştirilmiş bir uygulama yazıldı. Geliştirme makinesinde (Apple Silicon)
karşılaştırıldığında **arama 1,4–2,5 kat hızlı** çıktı ve vektörleştirme geri alındı.

**Hatanın fark edilmesi.** Ölçüm hedef donanımda yapılmamıştı. Karşılaştırma Jetson
üzerinde 220 gerçek 64×128 maske ile tekrarlandığında sonuç tersine döndü:
vektörleştirme **1,59 kat** hızlıydı (3,154 ms → 1,989 ms). Kare başına on bir çağrı
üzerinden bu, 34,7 ms'ye karşı 21,9 ms demektir.

**Neden ters döndü.** ARM tabanlı işlemcide Python yorumlayıcısı, NumPy'nin vektörleştirilmiş
işlemlerine göre çok daha yavaş çalışmaktadır. Yorumlayıcı ağırlıklı bir algoritma ile dizi
ağırlıklı bir algoritmanın karşılaştırması, bu orana doğrudan bağlıdır.

> **Genellenebilir sonuç:** uç cihaz optimizasyon kararları geliştirme makinesinde
> alınamaz. Aynı iki uygulama, aynı girdilerle, iki makinede zıt sonuç vermiştir.

**Üçüncü iyileştirme.** Aynı inceleme sırasında, on bir ayrı sınıf için yapılan on bir ayrı
tam dizi geçişinin tek bir geçişe indirilebileceği görüldü: farklı sınıflara ait bileşenler
birbirine bağlanamayacağı için yayılma aynı sınıftan komşularla sınırlandırıldığında bütün
sınıflar tek geçişte çözülmektedir. Bu, 15,45 ms → 3,96 ms ile **3,90 kat** kazanç
sağlamıştır.

## 4.5 SONUÇ

Şekil 4.2, optimizasyon öncesi ve sonrası kare bütçesini karşılaştırmalı olarak
göstermektedir.

**[Şekil 4.2: 07_frame_budget.pdf]**

Şekil 4.2. Optimizasyon öncesi ve sonrası kare bütçesi.

Çizelge 4.2. Optimizasyonun toplam etkisi (PIDNet-S).

| Ölçüt | Önce | Sonra | Kazanç |
|---|---|---|---|
| Kare süresi | 151,67 ms | 80,77 ms | **1,88×** |
| Kare başına enerji | 1,084 J | 0,665 J | **1,63×** |
| TensorRT motoru | değişmedi | değişmedi | — |
| Çıktılar | — | **birebir aynı** | — |

Elde edilen kazancın niteliği önemlidir:

- **Modele dokunulmamıştır.** Ağırlıklar, mimari ve TensorRT motoru değişmemiştir.
- **Kuantalama yapılmamıştır.** INT8'e inmek gibi doğruluk riski taşıyan bir yol
  izlenmemiştir.
- **Çıktılar birebir aynıdır.** Her iki optimizasyon da, eski uygulamanın test içinde
  saklanıp yeni uygulamayla karşılaştırıldığı diferansiyel testlerle doğrulanmıştır.

Yani hızlanma bir ödünleşimin sonucu değildir; ölçüm, maliyetin sanıldığı yerde olmadığını
göstermiş ve doğru yere yapılan müdahale bedelsiz kazanç sağlamıştır.

## 4.6 ULAŞILAMAYAN HEDEF

Optimizasyona rağmen gerçek zaman hedefi (≥ 20 FPS) **hiçbir yapılandırmada
karşılanamamıştır**; en iyi sonuç 12,36 FPS'tir. Bu, raporun Bölüm 6'da açıkça belirtilen
sınırlarındandır.

Bununla birlikte, darboğazın nerede olduğu artık ölçülmüş durumdadır: motorun payı %5,4,
geri kalanı CPU tarafıdır. Bu, hedefe giden yolun model değiştirmekten değil, CPU tarafını
azaltmaktan geçtiğini göstermektedir. Bölüm 5.7'de görüleceği üzere, aynı ölçüm bir
mimarinin çıktı çözünürlüğünün sistem maliyetini nasıl belirlediğini de ortaya
çıkarmaktadır.
