# Tez bölümleri — rubrikte açık kalan dördü

Bu dosya, `THESIS_OUTLINE.md`'de başlık olarak duran ama yazılmamış dört bölümü yazılmış
hâlde tutar: standartlara atıf, girişimcilik, sürdürülebilirlik, özet/abstract. Şablona
doğrudan yapıştırılabilecek olgunlukta yazılmıştır.

**Kural:** buradaki her sayı `THESIS_EVIDENCE.md`'deki ölçülmüş bir kayda dayanır. Hiçbir
cümle "beklenen" bir değeri "ölçülmüş" gibi sunmaz, hiçbir standart bu projeye
uygulanmadığı hâlde uygulanmış gibi anılmaz.

---

## C4 · Standartlara atıf (rubrik 5 puan)

Bu bölümün riski açıktır: standart adlarını sıralamak kolay, onlara gerçekten uyduğunu
iddia etmek yanlıştır. Aşağıdaki her madde, standardın **hangi maddesinin bu projede
karşılığı olduğunu** ve **nerede karşılığı olmadığını** ayrı ayrı söyler.

### C4.1 · ISO/IEC 25010 — Yazılım ürün kalite modeli

25010, yazılım kalitesini sekiz karakteristiğe ayırır. Bu projede dördü doğrudan ölçüldü,
dördü ölçülmedi ve ölçülmediği yazılmalıdır.

| 25010 karakteristiği | bu projedeki karşılığı | durum |
|---|---|---|
| **Performans verimliliği** — zaman davranışı | uçtan uca gecikme p50/p95/p99, sürdürülen FPS, 600 sn yük | **ölçüldü** |
| **Performans verimliliği** — kaynak kullanımı | tepe RAM (2.530–2.632 MiB), ortalama güç, joule/kare | **ölçüldü** |
| **Güvenilirlik** — olgunluk | 600 saniyelik sürdürülen yük altında termal kısma yok (GPU 56–59 °C) | **ölçüldü** |
| **Taşınabilirlik** — uyarlanabilirlik | PyTorch → ONNX → TensorRT FP16 zinciri, altın girdi/çıktı doğrulamasıyla | **ölçüldü** |
| İşlevsel uygunluk — doğruluk | mIoU, sınıf bazlı IoU, bileşen kapsama | **ölçüldü** |
| Kullanılabilirlik | — | ölçülmedi (arayüz sunum panelidir, kullanıcı çalışması yapılmadı) |
| Güvenlik (security) | — | ölçülmedi, kapsam dışı |
| Bakım yapılabilirlik | 702 birim testi, her hata için regresyon testi; ölçülmüş bir metrik değil | kısmi |

> **Dürüstlük notu:** 25010 bir *kalite modelidir*, uygunluk sertifikası değildir. Bu tez
> "ISO/IEC 25010 uyumludur" demez; 25010'un tanımladığı karakteristiklerden dördünü
> ölçtüğünü, dördünü ölçmediğini söyler.

### C4.2 · ISO/IEC/IEEE 29119 — Yazılım testi

**Önce bir künye düzeltmesi:** IEEE 829-2008 test dokümantasyon standardının yerini
ISO/IEC/IEEE 29119-3:2013 almış, o da 29119-3:2021 ile güncellenmiştir. Bu tezde IEEE
829'a atıf yapılmayacak; güncel atıf 29119'adır. Bu projedeki karşılığı:

- **Test tasarım teknikleri (29119-4):** hata tabanlı test. Karşılaşılan her gerçek hata
  için bir regresyon testi yazıldı — `Runner.from_cfg()`'nin `load_from`'u yüklememesi,
  `BoundaryLoss` bf16 çakışması, `np.str_` kırpması, AppleDouble yan dosyaları,
  `rglob`'un etiket klasörüne inmesi. Testler hatanın *kendisini* değil, hatanın
  *koşulunu* sabitler.
- **Test dokümantasyonu (29119-3):** her ölçüm kaydı kendi `record_type`,
  `schema_version`, `model_sha256`, tohum ve `scientific_measurement` alanını taşır; bir
  sonucun hangi kodla ve hangi ağırlıkla üretildiği kayıttan doğrulanabilir.
- **Karşılanmayan:** biçimsel test planı ve test tamamlama raporu üretilmemiştir. Bu bir
  bitirme projesidir, sertifikasyon süreci değildir.

### C4.3 · ISO 21448 (SOTIF) — Amaçlanan işlevin güvenliği

Bu, tezin problem tanımıyla **doğrudan örtüşen** standarttır ve bölümün en önemli
maddesidir. SOTIF, sistemin *arızalanmadığı* ama *yetersiz kaldığı* durumları ele alır:
donanım bozulmadan, yazılım çökmeden, sadece algı yeterli olmadığı için ortaya çıkan
tehlikeleri.

Bu tezin ölçtüğü şeylerin çoğu SOTIF sınıfı bulgulardır:

- **Bilinmeyen-tehlikeli senaryo:** eğitim dağılımında olmayan yol engelleri. Ölçüldü —
  RoadAnomaly'de kayıp yük (lost cargo) AUROC **0,4805**, yani rastgeleden kötü. Sistem
  arızalı değil; yola düşmüş yükü *fark edemiyor*.
- **Çalışma alanı sınırı (operational design domain):** gece. Ölçüldü — DDRNet-23-slim
  gece doğruluğunun yalnızca %10,4'ünü koruyor (7,12 mIoU) ve kalibrasyon hatası 14 kat
  büyüyor. Bu, ODD'nin gündüzle sınırlı olması gerektiğinin ölçülmüş gerekçesidir.
- **Aşırı güven:** model kesinlikle yanıldığı piksellerde %73–84 güven veriyor; ince
  yapılarda sınıf bazlı ECE havuzlanmış değerin 8 katına çıkıyor (`pole`, PIDNet-S,
  0,2598). SOTIF açısından tehlikeli olan yanılmak değil, **yanıldığını bilmemektir.**

> **Sınır — bu çok önemli:** proje SOTIF *sürecini* uygulamamıştır (tehlike analizi,
> senaryo kataloğu, kabul kriteri türetimi, doğrulama kampanyası). SOTIF burada, ölçülen
> olguları adlandıran bir **kavramsal çerçeve** olarak kullanılır. Tez "SOTIF uyumlu bir
> sistem" iddiasında bulunmaz ve bulunamaz.

### C4.4 · ISO 26262 — Yol araçları işlevsel güvenlik

Anılır ve **kapsam dışı bırakılır.** 26262 donanım/yazılım arızalarını ele alır ve ASIL
seviyesi, geliştirme süreci, araç entegrasyonu ve doğrulama kampanyası gerektirir. Bu
proje bir araştırma prototipidir; hiçbir ASIL seviyesi hedeflenmemiş, hiçbir güvenlik
gereksinimi türetilmemiştir. Bunun açıkça yazılması, mühendislik etiği açısından
standardı anmaktan daha değerlidir.

### C4.5 · ISO 690 — Kaynak gösterimi

Kaynakça ISO 690 biçiminde verilmiştir (`docs/BIBLIOGRAPHY.md`: 14 bilimsel kaynak +
5 standart, tamamı künyesi doğrulanmış).
Uygulanan kural: **listeye yalnızca birincil kaynağından açılıp künyesi doğrulanmış kayıt
girer.** Aday havuzu (912 kayıt) kaynakça değildir.

---

## C5 · Girişimcilik ve yenilikçilik (rubrik 4 puan)

### C5.1 · Çözülen gerçek problem

Uç cihaza model seçen bir ekibin bugün elindeki bilgi, model zoo'ların yayınladığı mIoU
sıralamasıdır. Bu tez o sıralamanın **dağıtım koşuluna taşınmadığını** ölçtü:
ρ(yayınlanmış, ölçülen) = **+0,10**. Yayınlanmış sıralamada sonuncu olan model, dağıtım
çözünürlüğünde sınıf-ortalamalı mIoU'da öne geçiyor.

Bunun pratik karşılığı doğrudan paraya çevrilebilir: yanlış model seçen bir ekip ya
gereksiz donanım alır ya da sahada yetersiz kalan bir sistem dağıtır. Seçim hatasının
maliyeti, seçimi doğru yapmanın maliyetinden büyüktür.

### C5.2 · Aktarılabilir yöntem

Bu tezin ürünü bir model değil, **bir seçim yöntemidir**:

1. Adayları **dağıtım çözünürlüğünde** ölç, yayınlanmış sayıya güvenme.
2. Farkların **gerçek olup olmadığını** test et — eşleştirilmiş bootstrap. Bu projede ilk
   iki modelin ayrışmadığı böyle bulundu (+0,0000 [−0,0050, +0,0049]).
3. Doğruluk ayrışmıyorsa seçimi **enerji ve dayanıklılığa** bırak.
4. Sistem maliyetini FLOP'tan değil **çıktı stride'ından** tahmin et.

Dört adımın hiçbiri bu veri kümesine veya bu donanıma özgü değildir. Uç cihaza
segmentasyon dağıtan herkes aynı sırayı uygulayabilir.

### C5.3 · Yenilikçi bulgu

**Sistem maliyetini segmentasyon başının çıktı stride'ı belirliyor, model boyutu değil.**
SegFormer-B0'ın TensorRT motoru DDRNet'ten yalnızca 11,2 ms yavaş; karesi 193,4 ms yavaş —
motor farkının **17 katı**. Sebep, stride-4 çıktısının CPU tarafındaki algı yığınına 4 kat
piksel vermesidir. Hızlandırıcı ölçeklenmiyor, CPU tarafı ölçekleniyor.

Bu bulgu literatürde yaygın olarak vurgulanmaz çünkü çoğu karşılaştırma **yalnızca motor
gecikmesini** raporlar. Uçtan uca ölçen bir çalışma bunu görmek zorundadır.

### C5.4 · Ürünleşme yolu ve dürüst engel

Prototipten ürüne giden yolda üç somut adım vardır: gece için ayrı bir çalışma kipi,
piksel-OOD için etiketli veri, ve FP16 dağıtım sadakatinin cihazda doğrulanması.

**Engel de açıkça yazılmalıdır:** hiçbir yapılandırmada gerçek zaman kapısı geçilmedi (en
iyi 12,36 FPS, hedef ≥ 20). Sistem bugünkü hâliyle ürün değildir. Darboğazın model değil
CPU tarafı olduğu ölçüldüğü için yol açıktır, ama yol henüz yürünmemiştir.

---

## C6 · Sürdürülebilirlik (rubrik 4 puan)

Bu bölümün gücü, genel ifadeler değil **doğrudan ölçülmüş joule** olmasıdır. Enerji,
Jetson modülünün dahili sensörlerinden `tegrastats` ile, 600 saniyelik sürdürülen yük
boyunca kaydedildi; kare başına enerji ortalama güçten ve sürdürülen FPS'ten türetildi.

### C6.1 · Mimari seçiminin enerji karşılığı

| model | J/kare | DDRNet'e göre |
|---|---|---|
| **DDRNet-23-slim** | **0,630** | — |
| PIDNet-S | 0,665 | +%5,6 |
| PIDNet-M | 0,808 | +%28,3 |
| BiSeNetV2 | 0,809 | +%28,4 |
| SegFormer-B0 | 2,221 | **+%252** |

DDRNet-23-slim'i PIDNet-M yerine seçmek kare başına **%22 enerji tasarrufu** sağlar
(0,808 → 0,630 J). Ve §9a'da ölçüldüğü üzere bu seçim **doğruluktan feragat değildir**:
ilk iki model kare düzeyinde ayrışmıyor.

> **Bölümün asıl cümlesi budur:** doğruluk ayrışmadığında enerji bedava bir tercih
> değildir — tek anlamlı tercihtir.

### C6.2 · Ölçeklendiğinde ne demek

Tek bir cihaz kesintisiz çalıştığında, 7,79 W'ta yılda **68,2 kWh**. Aynı kare sayısını
SegFormer-B0 ile işlemek **240,5 kWh** gerektirir — 3,53 katı. Bin cihazlık bir filoda
aradaki fark yılda **172 MWh** olur.

*(Bu bir projeksiyondur, ölçüm değildir: tek cihazın ölçülmüş joule/kare değeri
çarpılmıştır. Filo ölçeğinde soğutma, boşta kalma ve iletişim maliyetleri hesaba
katılmamıştır.)*

### C6.3 · Hesaplama bütçesinin kendisi

Sürdürülebilirlik yalnızca dağıtımın değil, geliştirmenin de meselesidir. Bu projede
bilinçli olarak yapılan üç şey:

- **Referans checkpoint'ler yeniden eğitilmedi.** mmsegmentation model zoo'daki
  Apache-2.0 ağırlıklar kullanıldı; beş mimarinin 120k–160k adımlık eğitimi
  tekrarlanmadı.
- **Kendi eğitimimiz 2.500 adımda sınırlandı** — yayınların %0,7'si kadar örnek. Bu bir
  eksiklik olarak da yazılmıştır; ama tekrar tekrar eğitmek yerine ölçüm derinliğine
  yatırım yapmak bilinçli bir karardı.
- **Optimizasyon modele değil kod tarafına yapıldı.** Kare 146,44 → 93,98 ms'ye indi,
  motor hiç değişmeden ve çıktılar birebir aynı kalarak. Yeniden eğitim gerektirmeyen
  hızlanma, en ucuz hızlanmadır.

### C6.4 · Ölçülmeyen

Eğitim aşamasının toplam enerjisi ölçülmedi (Colab bunu raporlamaz). Cihazın üretim ve
imha ayak izi kapsam dışıdır. Bu bölüm **kullanım aşaması** enerjisiyle sınırlıdır ve
öyle olduğu yazılmalıdır.

---

## C7 · ÖZET ve ABSTRACT

### ÖZET

Otonom sürüş algı sistemleri, eğitim sırasında görmedikleri nesnelerle karşılaştıklarında
sessizce yanılır: bir tahmin üretirler ve o tahminden emin olurlar. Bu tez, kaynak
kısıtlı bir uç cihazda çalışan, belirsizliğinin farkında ve açık kümeye dayanıklı bir yol
tehlikesi algılama hattı tasarlamakta ve ölçmektedir.

Beş gerçek zamanlı semantik bölütleme mimarisi (PIDNet-M/S, DDRNet-23-slim, SegFormer-B0,
BiSeNetV2) ortak bir protokolle, dağıtım çözünürlüğünde ve aynı karelerde karşılaştırılmış;
sistem NVIDIA Jetson Orin Nano Super üzerinde 25 W güç modunda, TensorRT FP16 ile, 600
saniyelik sürdürülen yük altında profillenmiştir. Belirsizlik dört skorla (MSP, entropi,
max-logit, energy) ölçülmüş, açık küme başarımı 60 kare piksel etiketli gerçek yol
tehlikesi üzerinde değerlendirilmiştir.

Dört bulgu elde edilmiştir. **Birincisi**, yayınlanmış model sıralaması dağıtım koşuluna
taşınmamaktadır (Spearman ρ = +0,10); uç cihaz için model seçimi yayınlanmış doğruluğa
bakılarak yapılamaz. **İkincisi**, sistem maliyetini model boyutu değil segmentasyon
başının çıktı stride'ı belirlemektedir: bir modelin hızlandırıcı üzerindeki gecikmesi 11
ms fazlayken uçtan uca karesi 193 ms fazladır, çünkü stride-4 çıktısı CPU tarafına dört
kat piksel vermektedir. **Üçüncüsü**, doğrulukta bir kazanan yoktur; eşleştirilmiş
karşılaştırma ilk iki modeli ayırt edememektedir (+0,0000, %95 aralık [−0,0050, +0,0049],
500 kare), dolayısıyla seçim enerji ve dayanıklılık eksenlerine kalmaktadır — ki bu iki
eksende kazananlar farklıdır. **Dördüncüsü**, havuzlanmış kalibrasyon hatası yanıltıcıdır:
sınıf bazlı hesaplandığında hata 1,5 ile 3 kat büyümekte ve en kötü sınıflar her modelde
ince yapılar (direk, çit) olmaktadır.

Sistem 12,36 FPS ile gerçek zaman hedefini (≥ 20 FPS) karşılamamaktadır; darboğazın model
değil CPU tarafındaki algı yığını olduğu ölçülmüş, kod tarafı optimizasyonuyla kare süresi
modele dokunulmadan 146,44 ms'den 93,98 ms'ye indirilmiştir.

**Anahtar kelimeler:** semantik bölütleme, açık küme tanıma, belirsizlik kestirimi,
kalibrasyon, uç bilişim, Jetson, TensorRT, otonom sürüş

### ABSTRACT

Autonomous driving perception systems fail silently when they encounter objects absent
from their training distribution: they produce a prediction, and they are confident in it.
This thesis designs and measures an uncertainty-aware, open-set road hazard detection
pipeline running on a resource-constrained edge device.

Five real-time semantic segmentation architectures (PIDNet-M/S, DDRNet-23-slim,
SegFormer-B0, BiSeNetV2) are compared under a common protocol, at deployment resolution
and on identical frames; the system is profiled on an NVIDIA Jetson Orin Nano Super in its
25 W power mode with TensorRT FP16 under a 600-second sustained load. Uncertainty is
quantified with four scores (MSP, entropy, max-logit, energy) and open-set performance is
evaluated on 60 pixel-labelled frames of real road hazards.

Four findings emerge. **First**, published model rankings do not transfer to the
deployment condition (Spearman ρ = +0.10); edge model selection cannot be made from
published accuracy. **Second**, system cost is determined by the segmentation head's
output stride rather than model size: one model's accelerator latency is 11 ms higher
while its end-to-end frame is 193 ms higher, because a stride-4 output feeds four times as
many pixels to the CPU-side stack. **Third**, there is no accuracy winner; a paired
comparison cannot separate the top two models (+0.0000, 95% interval [−0.0050, +0.0049],
500 frames), leaving the choice to energy and robustness — axes with different winners.
**Fourth**, pooled calibration error is misleading: computed per class it grows by a
factor of 1.5 to 3, and the worst classes in every model are thin structures (pole,
fence).

The system reaches 12.36 FPS and does not meet the real-time target of ≥ 20 FPS; the
bottleneck was measured to be the CPU-side perception stack rather than the model, and
code-side optimisation reduced frame time from 146.44 ms to 93.98 ms without touching the
network.

**Keywords:** semantic segmentation, open-set recognition, uncertainty estimation,
calibration, edge computing, Jetson, TensorRT, autonomous driving
