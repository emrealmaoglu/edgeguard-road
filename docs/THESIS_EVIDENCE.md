# Tez kanıt envanteri

Özetteki her iddianın karşısına, o iddiayı destekleyen **ölçülmüş** kanıt ve kaydın
bulunduğu dosya yazılmıştır. Ölçülmemiş olanlar açıkça boşluk olarak işaretlidir; hiçbir
satır "beklenen" değeri "ölçülmüş" gibi göstermez.

Bütün sayılar 2026-08-15 itibarıyla gerçek koşulardan gelir. Donanım: NVIDIA Jetson Orin
Nano Super, 25 W güç modu, TensorRT 10.3.0 FP16, JetPack 6 (L4T 36, aarch64).

---

## 1 · Modeller ve doğruluk

**İddia:** *"hafif derin öğrenme modellerini geliştirerek ve karşılaştırarak"*

İki ayrı kanıt hattı var ve karıştırılmamalıdır.

### 1a · Kendi eğitilen modeller (deneysel katkı)

Cityscapes + IDD20K karışık, 2.500 adım, batch 4, 512×1024 crop, bf16, ImageNet
başlangıcı, `seed 20260728`, NVIDIA L4.

| model | domain-macro mIoU | Cityscapes | IDD20K | nadir sınıf mIoU |
|---|---|---|---|---|
| pidnet_s | 0,3700 | 0,4641 | 0,2759 | 0,1755 |
| ddrnet_23_slim | 0,3574 | 0,4310 | 0,2840 | 0,1346 |
| segformer_b0 | 0,2286 | 0,2723 | 0,1849 | **0,0000** |

Kaynak: `reports/screening/candidate_table.json`. Üçü de `onnx_validated: true`,
`rejected: []`.

**Sınıf bazlı IoU** eğitim loglarında tam tablo hâlinde mevcut
(`screening/<model>/ce/<zaman>/<zaman>.log`, "per class results"). ddrnet_23_slim örneği:
road 85,98 · sidewalk 50,22 · building 58,24 · sky 70,82 · car 58,67 · vegetation 62,58
· person 32,67 · bicycle 35,15 · bus 32,64 · wall 32,18 · motorcycle 24,98 · traffic sign
21,88 · rider 19,49 · truck 18,29 · fence 12,93 · traffic light 11,44 · pole 7,45 ·
**train 1,33**.

**Nadir sınıf bulgusu:** segformer_b0'ın nadir sınıf mIoU'su tam olarak sıfır. 2.500
adımlık bütçede nadir sınıfların hiçbirini öğrenememiş. Sınıf dengesizliğinin mimariye
göre farklı vurduğunun somut kanıtı.

### 1b · Referans checkpoint'ler (karşılaştırma temeli)

mmsegmentation model zoo, Cityscapes'te 120k–160k adım eğitilmiş, Apache-2.0.
Kendi eğitimimiz bunların **%0,7'si kadar** örnek gördü (10.000'e karşı 1.440.000).

| model | mIoU (Cityscapes val) |
|---|---|
| PIDNet-M | 80,22 |
| PIDNet-S | 78,74 |
| DDRNet-23-slim | 77,84 |
| SegFormer-B0 | 76,54 |
| BiSeNetV2 | 75,76 |

**Geçersiz sayılar:** eski kampanyanın 16,48 / 20,53 / 26,54 / 25,21 / 17,41 değerleri
kullanılmayacak. O değerlendirme rastgele ağırlık ölçüyordu — `Runner.from_cfg()`
`cfg.load_from`'u yüklemez, bu yalnızca `train()/val()/test()` içinde olur. Hata
`load_evaluation_weights()` ile düzeltildi.

---

## 2 · Sınıf algılama ve konumlandırma

**İddia:** *"yol, kaldırım, araç, yaya, bisiklet, trafik işareti ve benzeri çevresel
sınıfları algılaması; önemli nesne ve bölgeleri görüntü üzerinde konumlandırması"*

19 Cityscapes sınıfı. Bölge çıkarımı `derive_perception` ile; her bölge için sınıf,
sınırlayıcı kutu, ağırlık merkezi, alan, ortalama güven, ortalama entropi, koridora
uzaklık ve piksel maskesi üretiliyor (`SemanticRegion`).

Kanıt: `predict.py --emit-regions` çıktıları, `results/figures/`.

---

## 2b · Nesne düzeyinde konumlandırma — piksellerin göremediği

**İddia:** *"sürülebilir alan ve potansiyel risk bölgelerini belirlemesi"* — belirlemek,
piksel saymaktan farklı bir iştir. mIoU büyük bir engelin üçte birini bulan modelle,
aynı engeli üç ayrı parça olarak bulan modeli aynı puanlar; küçük ama ölümcül bir engeli
tamamen kaçıranı da neredeyse cezalandırmaz. Araştırma dokümanı 26 bunu söylüyor ve
bileşen düzeyinde metrik istiyor. `component_localization_metrics` tam bunun için
yazılmıştı — testli, çağrısız (`drivable_metrics` ve `paired_comparison` ile aynı örüntü).

100 Cityscapes val karesi, on dikkat sınıfı, minimum 64 piksel
(`scripts/evaluate_component_localization.py`):

| mimari | bileşen kapsama | en iyi bileşen IoU | tahmin/GT bileşen | yorum |
|---|---|---|---|---|
| SegFormer-B0 | **0,6780** | 0,4351 | 42,50 / 27,86 = **1,53×** | daha çok buluyor, daha çok parçalıyor |
| PIDNet-S | 0,6655 | 0,4456 | 27,81 / 27,86 = 1,00× | sayıca denk |
| DDRNet-23-slim | 0,6428 | **0,4630** | 23,94 / 27,86 = **0,86×** | daha az buluyor, bulduğunu daha bütün buluyor |

> **Stride-4 hikâyesinin dördüncü kanıtı.** SegFormer gerçek nesnelerin daha büyük bir
> kısmını buluyor (kapsama 0,678'e karşı 0,643) ama onları **1,53 kat fazla parçaya**
> bölüyor (1,5255×). DDRNet'inki tersi: daha az nesne buluyor, ama bulduğunun bileşen IoU'su en
> yüksek (0,463). İnce ızgara ayırıyor, kaba ızgara birleştiriyor.

**Bu neden sadece bir metrik değil.** Zamansal izleyici ve risk sıralaması **bileşenler
üzerinde** çalışır. İkiye bölünmüş bir nesne, dikkat için yarışan iki izdir — yani
parçalanma aşağı akışta doğrudan §3b'deki titremeye dönüşür. SegFormer'ın sürülebilir alan
ölçümünde de en parçalı yol maskesini vermesi (8,62'ye karşı 4,54) aynı olgudur.

**Sınır:** bunlar sınıf-bağlantılı bölgelerdir, **örnek (instance) değildir** — yan yana
duran aynı sınıftan iki araba burada tek bileşendir. Kayıt `instance_detection: false`
diyor; bu bir dedektör kıyaslaması değildir.

---

## 2b-2 · Rakip açıklama: nadirlik mi, incelik mi?

Bu tezin dört ölçümü aynı sınıflara işaret ediyor — direk, çit, trafik ışığı, insan — ve
şimdiye kadarki açıklaması **çıktı stride'ı**: kaba ızgara ince yapıyı kötü çözüyor. Bu bir
mekanizma, ama aynı kurbanları öngören **ikinci bir mekanizma** var ve ölçülmeden
elenemez: o sınıflar aynı zamanda nadir sınıflardır, model onların daha az pikselini
görür. Biri iddia edilip diğeri ölçülmeden bırakılamaz.

500 Cityscapes val karesi, 917 milyon etiketli piksel, yalnızca zemin gerçeği
(`scripts/measure_class_distribution.py`):

**Dengesizlik 473 kat** — `road` %37,65, `motorcycle` %0,080.

| ilişki | Spearman ρ | okuma |
|---|---|---|
| piksel payı ↔ sınıf IoU | **+0,68** | nadir sınıf daha kötü segmentleniyor |
| piksel payı ↔ sınıf ECE | **−0,52** | nadir sınıf daha kötü kalibre |

Yani **nadirlik gerçek bir etken.** Ama tek etken değil, ve bunu gösteren şey sıralamanın
kendisi değil, içindeki **ayrışma**:

| sınıf | piksel payı | karelerde | ECE (PIDNet-S) | şekil |
|---|---|---|---|---|
| `motorcycle` | **%0,080** (en nadir) | %18,6 | **0,0288** (en iyilerden) | kompakt |
| `train` | %0,113 | %4,4 | 0,0510 | kompakt |
| `truck` | %0,301 | %16,0 | 0,0345 | kompakt |
| `traffic light` | %0,197 | %58,2 | **0,1499** | **ince** |
| `rider` | %0,215 | %50,6 | **0,1711** | **ince** |
| `pole` | **%1,479** (7. en sık) | **%98,2** | **0,2598** (en kötü) | **ince** |

> **Ayrışma:** `pole`, `motorcycle`'dan **18 kat daha sık** ve karelerin neredeyse
> tamamında var — buna rağmen PIDNet-S'in kalibrasyon hatası orada **9 kat daha kötü**.
> Eşleşmiş nadirlikte de aynı: `truck` (%0,301) ile `traffic light` (%0,197) benzer
> nadirlikte, ama ECE'leri 0,0345'e karşı 0,1499 — **4,3 kat** fark. Ayıran şey nadirlik
> değil, **şekil**.

**Sonuç — ve bu bir tez cümlesidir:** sınıf dengesizliği başarımı etkiliyor (ρ = +0,68),
ama bu projedeki başarısızlık örüntüsünü açıklamıyor. İnce ve uzun yapılar, **aynı
nadirlikteki kompakt sınıflardan sistematik olarak daha kötü** segmentleniyor ve daha kötü
kalibre ediliyor. Kaba çıktı ızgarası bunu doğrudan öngörür; nadirlik öngörmez.

**Sınır:** bu bir gözlemsel ayrışmadır, kontrollü bir deney değil. "İncelik" burada
niceliksel bir değişken değil, sınıfların bilinen geometrisidir. Kesin kanıt, aynı sınıfın
farklı stride'larda ölçülmesi olurdu — §8'de tek mimari için (SegFormer, 2,03 mIoU bedeli)
yapıldı, beş mimari için yapılmadı.

---

## 2c · Split sızıntısı — her doğruluk sayısının dayandığı varsayım

Araştırma dokümanı 18 bu konuda net: sürüş görüntüsü 15-30 FPS'te kaydedilir, saniyenin
üçte biri arayla iki kare neredeyse aynıdır, ve rastgele bölme birini eğitime diğerini
doğrulamaya atarsa model segmentlemeyi değil ezberlediği arka planı tanımayı öğrenir.
Bu tezdeki her doğruluk sayısı, bunun olmadığı varsayımına dayanıyor —
**ve varsayım bugüne kadar test edilmemişti.** `bounded_perceptual_duplicate_pairs` bu iş
için yazılmış, çağrılmamıştı.

446 kare, dört değerlendirme kümesi, 64-bit ortalama-hash
(`scripts/audit_split_leakage.py`):

| yarıçap | **splitler arası** | cityscapes_val | demo_video | acdc_night | acdc_fog |
|---|---|---|---|---|---|
| d ≤ 0 | **0** | 0 | 36 | 0 | 2 |
| d ≤ 2 | **0** | 0 | 127 | 1 | 88 |
| d ≤ 4 | 7 | 2 | 293 | 12 | 581 |
| d ≤ 6 | 90 | 16 | 516 | 80 | 1.504 |

> **Sızıntı yok.** Splitler arası yakın-kopya sayısı d≤2'de **sıfır**. Gerçek bir kopya
> d=0'da görünür ve yarıçap büyüdükçe orada kalır; buradaki sayı 0 → 0 → 7 → 90 diye
> *sadece yarıçapla* büyüyor, ki bu kopya değil, hash'in sinyali tükenmesidir.
> Cityscapes val kendi içinde de temiz: 120 eşit aralıklı karede d≤2'de **sıfır**
> yakın-kopya. Manşet doğruluk sayıları tekrar eden sahnelerden şişmiş değil.

**Yöntemsel bulgu — dokümanın öngörmediği.** Aynı yarıçap her görüntüde aynı şeyi
ölçmüyor. ACDC sis kümesi d=0'da 2 çiftten d=6'da 1.504'e çıkıyor (**752 kat**), Cityscapes
val ise 0'dan 16'ya. Sis kontrastı yok ediyor, ortalama-hash ayırt etme gücünü kaybediyor.
**Sabit bir algısal-hash eşiği hava koşulları arasında taşınamaz** — olumsuz koşul
verisinde dedup yapan bir çalışma bunu hesaba katmak zorundadır. Araştırma dokümanı 18
algısal dedup öneriyor ama bu sınırı belirtmiyor; biz ölçtük.

demo_video'nun d=0'da bile 36 çift vermesi **beklenen ve doğru**: o bitişik bir video
dizisidir, ardışık kareler gerçekten yakın-kopyadır.

**Sınır:** algısal hash kimlik kanıtı değildir (`identity_proof: false`). İki karenin
farklı sahneler olduğunu kanıtlamaz, yalnızca bu yarıçapta yakın-kopya olmadıklarını.

---

## 3 · Sürülebilir alan ve risk bölgeleri

**İddia:** *"sürülebilir alan ve potansiyel risk bölgelerini belirlemesi"*

- **Sürülebilir alan:** `drivable_corridor_from_semantics` — yol pikselleri arasından
  alt-orta bölgeye bağlı bileşen seçiliyor (`road_mask.png`, `drivable_corridor.png`).
- **Risk bölgeleri:** yedi özellikli açıklanabilir füzyon (`contextual_risk`).

### 3a · Sürülebilir alan — sayısal ölçüm (Cityscapes val, 200 kare, 5 mimari)

Bu bölüm bugüne kadar yalnızca **görsele** dayanıyordu. `evaluation/perception.py`
içindeki metrik takımı baştan beri hazır ve testliydi ama çağıranı yoktu; ihtiyaç duyduğu
şey gerçek yol maskeleriydi ve Cityscapes val onları veriyor. Ölçüm aşağıda.

| mimari | yol IoU | sınır F1 (1 px) | sınır F1 (8 px) | **yanlış-sürülebilir** | yol parçası |
|---|---|---|---|---|---|
| SegFormer-B0 | **0,9653** | **0,2336** | **0,6143** | 0,0091 | 8,62 |
| DDRNet-23-slim | 0,9622 | 0,1683 | 0,5817 | **0,0064** | **4,54** |
| PIDNet-M | 0,9617 | 0,1581 | 0,5651 | 0,0065 | 4,92 |
| PIDNet-S | 0,9614 | 0,1604 | 0,5651 | 0,0076 | 4,63 |
| BiSeNetV2 | 0,9602 | 0,1717 | 0,5910 | 0,0097 | 5,25 |

Kayıt: `results/drivable/*.json`. Ignore pikselleri (255) dışlanmıştır.

**Yanlış-sürülebilir oranı**, aracın gireceği ama yol *olmayan* piksellerin oranıdır —
güvenlik açısından anlamlı olan sayı budur. IoU'nun tamamı 0,5 puanlık bir aralığa
sıkışırken (yol kolay bir sınıftır) yanlış-sürülebilir 0,0064–0,0097 arasında **1,5 kat**
değişiyor. Yani mimarileri ayıran metrik IoU değil, bu.

**İki tolerans, tek sebep.** 1 piksel toleransı, kaba logit'ten büyütülmüş bir maskeden
çözünürlüğünün izin vermediği bir kesinlik ister; tek başına raporlansaydı modelin
başarısızlığı gibi okunurdu, oysa bir çözünürlük sınırıdır. 8 piksel **her mimari için
aynı** tutuldu: beşten dördünün dağıtım logit stride'ı budur. Tolerans modele göre
esnetilseydi aşağıdaki bulgu metriğin içinde kaybolurdu.

**Bulgu — stride 4'ün iki yüzü.** SegFormer-B0 logit'lerini stride 4'te (128×256), diğer
dördü stride 8'de (64×128) üretir (`logit_stride_tradeoff.json`). Sınır F1'de tek başına
öne çıkması (0,2336'ya karşı 0,158–0,172, **%36–48 daha iyi**) tesadüf değil, tam olarak
bunun sonucudur. Bedeli de aynı ölçümde görünüyor: yol maskesi **1,9 kat** daha parçalı
(8,62'ye karşı 4,54–5,25) ve yanlış-sürülebilir oranı en kötü ikinci. Daha ince ızgara
daha keskin sınır çiziyor, ama aynı incelik sahte küçük yol lekelerini de geçiriyor.
Üçüncü bir bedel zaten ölçülmüştü: aynı stride farkı post-processing'i 3,98 kat
pahalılaştırıyor ve logit'i stride 8'e indirmek 2,03 mIoU'ya mal oluyor. Tek mimari
seçimi, üç ayrı ölçümde tutarlı biçimde ortaya çıkıyor.

**Bulgu — ego koridoru adımının ölçülmüş karşılığı.** Ham yol maskesinden ego koridorunu
ayırmak bugüne kadar bir tasarım tercihiydi; ne kazandırdığı ölçülmemişti. Beş mimaride
de aynı yönde çıkıyor:

| mimari | yanlış-sürülebilir: yol → koridor | değişim | IoU bedeli |
|---|---|---|---|
| PIDNet-M | 0,0065 → 0,0048 | **−%26,2** | −0,53 puan |
| PIDNet-S | 0,0076 → 0,0058 | **−%23,7** | −0,57 puan |
| SegFormer-B0 | 0,0091 → 0,0071 | **−%22,0** | −0,53 puan |
| BiSeNetV2 | 0,0097 → 0,0077 | **−%20,6** | −0,67 puan |
| DDRNet-23-slim | 0,0064 → 0,0052 | **−%18,8** | −0,54 puan |

Koridor seçimi, ego konumuna bağlı olmayan yol bileşenlerini atar — yani tam olarak sahte
yol lekelerini. Beş mimaride de **yaklaşık 0,55 puan IoU karşılığında yanlış-sürülebilir
piksellerin beşte biriyle dörtte biri arası** eleniyor. Güvenlik açısından bu takas
doğru yönde: kaybedilen şey doğru yolun bir kısmı, kazanılan şey yanlış yola girmemek.

Gerçek tehlike karelerinde ölçülmüş sıralama (PIDNet-S referans, RoadAnomaly):

| kare | 1. sıra | risk | seviye | baskın etken |
|---|---|---|---|---|
| animals19_Porte_de_Roubaix | bicycle | 0,813 | high | anomaly_score |
| animals06_sheep_roads_lambs | person | 0,806 | high | anomaly_score |
| obstacles12_rocks5 | rider | 0,819 | high | anomaly_score |
| frankfurt_000000_002196 (temiz şehir) | person | 0,885 | high | anomaly_score |

Kayıt: `results/risk/*.json`.

### 3b · Zamansal kalıcılık — yedinci özelliğin ölçümü

`temporal_persistence`, risk sözleşmesinin yedinci özelliği ve bugüne kadar **hiç ağırlık
taşımamış** olanı. Tek kare onu üretemez, o yüzden sıfır *ağırlıkla* dışlanıyordu — doğru
bir karar, ama yedi özellikli bir füzyon iddia edip yedisinden birini hiç koşmamış olmak
başka bir sorun. Ardışık kare bunu çözer.

150 ardışık kare (Cityscapes demoVideo `stuttgart_00`), PIDNet-S referans, bölgeler
`TemporalPersistence` ile ilişkilendirildi — tek-kare sürücüsünün düştüğü en-yakın-merkez
yaklaşımı değil, gerçek izleyici.

| ölçüm | değer |
|---|---|
| bölge gözlemi | 5.208 |
| iz (track) | 1.441 (adil değerlendirilen: 1.382) |
| **tek karelik iz** | **692 — %50,1** |
| ortalama iz ömrü | 3,68 kare |
| **medyan iz ömrü** | **1 kare** |
| 1. sıradaki bölgesi değişen kare | 30 / 150 — **%20,0** |
| 3+ kare yaşayan iz | 471 |
| **titreyen iz** (kategori ≥2 kez değişen) | **89 — %18,9** |
| kararlı iz başına ort. kategori değişimi | 0,79 |
| high → daha düşük gözlem | 140 |
| daha düşük → high gözlem | 162 |

Kayıt: `results/temporal/pidnet_s.json`.

> **Bulgu: sistemin işaretlediği şeyin yarısı tek kare yaşıyor.** Medyan iz ömrü **1
> karedir**. Yani tek-kare operasyonel dikkat, baskın davranışı olarak titriyor. Zamansal
> kalıcılık bu tabloda "olsa iyi olur" bir özellik değil, birincil hata kipine denk gelen
> özelliktir.

**İkinci bir bozulma kipi: titreme.** İz ömrü ile titreme aynı şey değildir ve
araştırma dokümanı 26 bunları ayrı tanımlar — haklı olarak. Üç kareden uzun yaşayan 471
izin **%18,9'u** risk kategorisini en az iki kez değiştiriyor. Bu, tek karelik izden farklı
ve tartışmalı biçimde daha kötü bir hata: kaybolan bir uyarı değil, **kendini sürekli
yalanlayan kalıcı bir uyarı**. Sürücü güvenini en çok sarsan kip budur.

**Mekanizma — neden gerçek bir değişiklik.** İki skorlama farklı ağırlık toplamlarıyla
normalize ediliyor: kalıcılık olmadan 0,90, kalıcılıkla 0,95. Bir kez görülmüş bölge payda
büyürken paya neredeyse hiçbir şey eklemez, yani **gerçekten düşer**; hayatta kalmış bölge
yeterince kazanıp yükselir. İki skorlama tek ağırlıkla yapılsaydı özellik yalnızca
ekleyebilirdi ve ölçüm hiçbir şey söylemezdi. Bu, `test_activating_persistence_demotes_a_
flicker_and_promotes_a_survivor` ile sabitlendi.

**Sınır — bunun ölçmediği şey.** demoVideo'nun etiketi yok. Dolayısıyla ölçülen şey
izlerin **geçici** olduğudur, **yanlış** olduğu değil. Gerçekten bir kare görünüp kaybolan
bir yaya ile bir karelik segmentasyon gürültüsü bu ölçümde ayırt edilemez. "Kalıcılık
yanlış alarmları eler" cümlesi bu veriyle **kurulamaz**; kurulabilen cümle şudur:
*tek-kare dikkat sıralamasının %20'si, hiçbir yeni görüntü bilgisi olmadan yalnızca zamana
bakılarak değişiyor.*

**Karar:** özellik dağıtımda hâlâ tek karede sıfır ağırlıklıdır — çünkü tek karede
ölçülemez, ve bu ölçüm onu değiştirmez. Değiştirdiği şey, dışlamanın *bedelinin* artık
bilinmesidir.

---

**Dürüstlük sınırı:** `detector_overlap` (nesne dedektörü yok) ve `temporal_persistence`
(tek kare) **sıfır ağırlıkla** dışlandı, sıfır değerle değil. Sıfır değer verilseydi
ölçülmemiş bir sinyal "risk yok" gibi görünür ve bütün skorları aşağı çekerdi. Kayıt
uygulanan ve dışlanan ağırlıkları ayrı ayrı yazıyor. Çıktı
`calibrated_physical_risk_probability: false` — bu bir operasyonel dikkat sıralamasıdır,
fiziksel risk olasılığı değil.

---

## 4 · Güven ve piksel düzeyinde belirsizlik

**İddia:** *"tahmin güvenini ve piksel düzeyindeki belirsizliği ölçmesi"*

Dört skor üretiliyor: maximum softmax probability, normalize entropi, maximum logit,
energy (`uncertainty_maps`).

### Kalibrasyon bulgusu

Anomali pikselleri 19 sınıfın hiçbiri değildir; model ne tahmin ederse etsin **kesinlikle
yanılır**. O piksellerdeki güveni:

| model | anomali px güven | normal px güven | oran |
|---|---|---|---|
| pidnet_s | 0,7850 | 0,8214 | 0,956× |
| pidnet_m | 0,7934 | 0,8397 | 0,945× |
| segformer_b0 | 0,8404 | 0,9106 | 0,923× |
| ddrnet_23_slim | 0,7310 | 0,8060 | 0,907× |
| bisenetv2 | 0,7304 | 0,8165 | 0,895× |

Model, **kesin yanıldığı yerlerde bile %73–84 güven** veriyor; güveni yalnızca %4,4–10,5
düşüyor. Ciddi aşırı-güven. PIDNet-S en aşırı güvenli, BiSeNetV2 en dürüst.

Bu, MSP'nin neden zayıf bir OOD skoru olduğunu (AUROC 0,569) ve energy'nin neden daha iyi
olduğunu (0,667) doğrudan açıklıyor.

### Havuzlanmış ECE neyi saklıyor — ölçüldü

Bu projede raporlanan bütün ECE değerleri **bütün pikselleri tek bir güven histogramında
havuzluyor**. Araştırma notlarımızdan 26 numaralı doküman bunun yetersiz olduğunu ve sınıf
bazlı ECE'nin zorunlu olduğunu söylüyor; gerekçesi de şu: sürüş sahnesi yol, bina ve
gökyüzü ile dominedir, model en çok onlarda emin ve haklıdır, dolayısıyla onların iyi
kalibre kütlesi kritik sınıflardaki aşırı güveni **soğurabilir**.

Bu bir gerekçe. Ölçtük (`scripts/measure_classwise_calibration.py`, 200 Cityscapes val
karesi, kare başına 20.000 piksel örneklemi, tohum 20260728):

| model | havuzlanmış ECE | **sınıf-bazlı ECE** | oran | en kötü sınıf |
|---|---|---|---|---|
| SegFormer-B0 | 0,0148 | **0,0446** | **3,01×** | fence 0,1223 |
| PIDNet-S | 0,0323 | **0,0776** | **2,40×** | pole **0,2598** |
| DDRNet-23-slim | 0,0409 | **0,0621** | **1,52×** | pole 0,1744 |

> **Havuzlanmış ECE her modelde iyimser — 1,5 ile 3 kat.** Kaynağı da doğrudan görünüyor:
> `road` piksellerin **%39'unu** kaplıyor ve her modelde en iyi kalibre sınıflardan biri
> (SegFormer'da ECE 0,0029). Havuzlanmış sayıyı taşıyan sınıf bu.

**Sıralama da değişiyor.** Havuzlanmış ECE'ye göre kalibrasyon sıralaması SegFormer-B0 →
PIDNet-S → DDRNet; sınıf-bazlıya göre SegFormer-B0 → **DDRNet → PIDNet-S**. Son iki model
yer değiştiriyor. Yani hangi ECE'yi raporladığınız, hangi modelin daha iyi kalibre olduğu
sorusunun cevabını değiştiriyor.

**En kötü sınıflar her modelde ince yapılar:** direk (`pole`) ve çit (`fence`). PIDNet-S
direkte 0,2598 ECE veriyor — kendi havuzlanmış değerinin **8 katı** — ve aşırı-güven
işareti pozitif (+0,2598), yani model yanıldığı yerde emin. `person` sınıfı da üç modelde
de pozitif aşırı-güvenli (+0,027 … +0,064).

> **Tez boyunca tekrar eden iplik.** İnce yapılar — direk, trafik ışığı, levha, çit,
> insan — üç bağımsız ölçümde de sistemin zayıf noktası: sınıf bazlı IoU'da (§9a),
> sınır F1'inde (§3a) ve şimdi kalibrasyonda. Ortak nedeni çıktı stride'ıdır: kaba ızgara,
> ince yapıyı hem yanlış segmentliyor hem de yanlışlığından emin oluyor. Bu, ayrı ayrı
> ölçülen üç sonucun tek bir mekanizmaya bağlandığı yerdir.

**Tanım sınırı:** buradaki sınıf-bazlı ECE, **her sınıf olarak tahmin edilen** piksellerin
top-1 güvenini o sınıf içinde binleyip sınıflara eşit ağırlık verir. Literatürdeki
one-vs-rest classwise-ECE tanımından farklıdır ve kayıt bunu `classwise_definition`
alanında yazar; iki tanım karıştırılmamalıdır. %1'den az piksel tahmin edilen sınıflar
raporlanır ama ortalamaya girmez — bin gürültüsü hâkim olurdu.

**Boşluk:** ECE / reliability diyagramı ölçülmedi. Etiketli Cityscapes validasyon verisi
gerektiriyor; `evaluate.py run --fit-temperature` altyapısı hazır ve rol kapısı
sağlanabilir durumda, veri temin edilince tek komutla koşar.

---

## 5 · Açık küme / yol tehlikesi algılama

**İddia:** *"eğitim dağılımından farklı ... koşulları fark edebilmesi ve güvenilir olmayan
durumları işaretleyebilmesi"*

**Veri:** RoadAnomaly (Lis ve ark., EPFL CVLab), 60 kare, piksel etiketli gerçek yol
tehlikeleri — hayvan, kaya, koni, enkaz. Değerlendirme amaçlı; hiçbir model bu veriyle
eğitilmedi veya ince ayarlanmadı (`model_trained_on_this_data: false`).

**Protokol:** kare başına 20.000 piksel deterministik alt-örnekleme (`seed 20260728`),
toplam 1.200.000 piksel, %9,8 anomali.

### Skor karşılaştırması (PIDNet-S referans)

| skor | AUROC | AP | FPR95 |
|---|---|---|---|
| energy | **0,6670** | **0,1562** | 0,8608 |
| maximum logit | 0,6617 | 0,1539 | 0,8674 |
| normalize entropi | 0,6022 | 0,1322 | 0,9399 |
| MSP | 0,5693 | 0,1163 | 0,9462 |

`energy > max-logit > entropi > MSP` sıralaması OOD literatürünün bildirdiği sıralamayla
birebir aynı — uygulamayı bu sayılardan bağımsız olarak doğruluyor.

Bootstrap %95 GA (energy, 200 yeniden örnekleme): AUROC [0,6655, 0,6684],
AP [0,1550, 0,1574], FPR95 [0,8581, 0,8638].

### Tehlike türüne göre (energy)

| tehlike | AUROC | AP | anomali piksel |
|---|---|---|---|
| araç | 0,7070 | 0,1689 | 6.260 |
| koni | 0,6884 | 0,0678 | 4.552 |
| hayvan | 0,6717 | 0,2128 | 78.711 |
| engel | 0,6276 | 0,1189 | 28.190 |
| **kayıp yük** | **0,4805** | 0,0062 | 419 |

**Güvenlik bulgusu:** model kayıp yükte (lost cargo) rastgeleden kötü — AUROC 0,48. Yol
üzerine düşmüş yükü fark edemiyor.

### Operasyonel eşikler (energy)

| politika | eşik | TPR | FPR | F1 |
|---|---|---|---|---|
| F1-optimal | −6,233 | 0,450 | 0,210 | 0,267 |
| %5 risk bütçesi | −5,013 | 0,077 | 0,050 | 0,100 |

---

## 6 · Hava ve ışık koşullarına dayanıklılık

**İddia:** *"eğitim dağılımından farklı hava, ışık, yol ve trafik koşullarını fark
edebilmesi"*

### 6a · Gerçek olumsuz koşullar (ACDC, 406 kare, 5 mimari)

| model | temiz | sis | kar | yağmur | **gece** | gece kaybı | gece ECE |
|---|---|---|---|---|---|---|---|
| **SegFormer-B0** | 69,34 | 59,93 | 46,33 | 45,71 | **20,59** | **−70,3%** | 0,3150 |
| PIDNet-S | 67,68 | 57,14 | 42,79 | 39,64 | 14,97 | −77,9% | 0,2694 |
| BiSeNetV2 | 66,02 | 46,59 | 36,82 | 37,75 | 13,34 | −79,8% | 0,2712 |
| PIDNet-M | 68,47 | 60,29 | 41,53 | 42,64 | 12,54 | −81,7% | 0,3812 |
| **DDRNet-23-slim** | 68,50 | 56,99 | 38,75 | 44,59 | **7,12** | **−89,6%** | **0,4626** |

**Üç bulgu:**

1. **Gece bütün mimarilerde felaket.** En iyisi bile doğruluğunun %70'ini kaybediyor.
   Kalibrasyon her modelde **8,7–14,0 kat** bozuluyor (ECE ~0,03 → 0,27–0,46).

2. **Ama mimariler eşit çökmüyor.** SegFormer-B0 gecede doğruluğunun %29,7'sini
   koruyor; DDRNet-23-slim yalnızca %10,4'ünü — yani gecede **işlevsiz** (7,12 mIoU).

3. **Hız/enerji kazananı, dayanıklılık kaybedeni.** DDRNet-23-slim en hızlı (77,43 ms)
   ve en verimli (0,630 J/kare) model; aynı zamanda olumsuz koşullara en kırılgan olanı.
   Bu, üç eksenli seçime **dördüncü bir ekseni** ekliyor.

### 6b · Birleşik örüntü: tanıdık olmayan girdiye dayanıklılık

SegFormer-B0 iki bağımsız "tanıdık olmayan girdi" testinde de birinci:

| test | SegFormer-B0 | DDRNet-23-slim |
|---|---|---|
| bilinmeyen **nesne** (açık küme AP) | **0,3469** | 0,1651 |
| bilinmeyen **koşul** (gecede korunan doğruluk) | **%29,7** | %10,4 |

Beşlideki tek transformer bu. Dikkat tabanlı küresel bağlamın, girdi dağılımı kaydığında
CNN'lerin yerel özniteliklerinden daha zarif bozulduğu yorumu bu iki ölçümle tutarlı —
ancak n=5 ile bu bir gözlemdir, kanıtlanmış mekanizma değil.

### 6c · Sentetik bozulma (karşılaştırma amaçlı)

15 RoadAnomaly karesi, `rescue.stress._corrupt`, severity 0,7, PIDNet-S:

| koşul | ort. entropi | artış | düşük-güven piksel |
|---|---|---|---|
| temiz | 0,1709 | 1,00× | 3,55% |
| sis | 0,2009 | 1,18× | 6,76% |
| kar | 0,1906 | 1,12× | 6,01% |
| yağmur | 0,1736 | 1,02× | 4,33% |
| gece | 0,1703 | **1,00×** | 3,95% |

**Sentetik ile gerçek arasındaki uçurum:** sentetik "gece" (parlaklık düşürme) belirsizlik
sinyalinde **hiçbir tepki** yaratmıyor; gerçek ACDC gecesinde doğruluk %78 düşüyor. Yani
sentetik bozulma, gerçek koşul kaymasının yerine geçemez — bu, sentetik stres testlerine
dayanan çalışmalar için doğrudan bir uyarıdır.

## 7 · Uç cihaz performansı

**İddia:** *"gecikme, bellek tüketimi, güç kullanımı ve gerçek zamanlı çalışma kapasitesi"*

Beş model, aynı donanım, aynı güç modu, aynı kareler. Her biri 600 saniye sürdürülen yük,
200 kare ısınma, tam telemetri.

| model | motor | kare | FPS | ort. güç | J/kare | tepe RAM | logit çıktısı |
|---|---|---|---|---|---|---|---|
| PIDNet-M | 11,15 ms | 88,50 ms | 10,82 | 8,75 W | 0,808 J | 2.632 MiB | 64×128 |
| PIDNet-S | 5,07 ms | 80,77 ms | 11,87 | 7,89 W | 0,665 J | 2.601 MiB | 64×128 |
| **DDRNet-23-slim** | **3,81 ms** | **77,43 ms** | **12,36** | **7,79 W** | **0,630 J** | 2.541 MiB | 64×128 |
| SegFormer-B0 | 15,04 ms | **270,82 ms** | **3,57** | 7,93 W | **2,221 J** | 2.605 MiB | **128×256** |
| BiSeNetV2 | 13,45 ms | 92,36 ms | 10,34 | 8,36 W | 0,809 J | 2.530 MiB | 64×128 |

Hiçbirinde termal kısma yok (GPU 56–59 °C). Güç 25 W bütçesinin üçte biri.

**Gerçek zaman kapısı hiçbirinde geçmedi** (medyan ≤ 50 ms, ≥ 20 FPS). Ama sebebi
model değil — bkz. §8.

---

## 8 · Kare bütçesinin nereye gittiği

Aşama profili (PIDNet-S, 60 kare, gerçek cihaz):

| aşama | ilk ölçüm | optimizasyon sonrası | pay |
|---|---|---|---|
| derive_perception | 96,05 ms | 43,64 ms | 46,4% |
| preprocess | 24,80 ms | 24,71 ms | 26,3% |
| görüntü çözme | 13,97 ms | 14,00 ms | 14,9% |
| confidence_entropy | 5,78 ms | 5,80 ms | 6,2% |
| **motor** | **5,09 ms** | **5,09 ms** | **5,4%** |
| argmax | 0,75 ms | 0,74 ms | 0,8% |
| **toplam** | **146,44 ms** | **93,98 ms** | |

**Ana bulgu:** TensorRT motoru kare bütçesinin yalnızca **%5,4'ü**. Gerçek zamanlılık
hızlandırıcıda değil, CPU tarafındaki algı yığınında kazanılıp kaybediliyor.

### İki hedefli optimizasyon, modele dokunmadan

1. `_distance_from_mask` piksel-piksel Python BFS kuyruğuydu. Engelsiz 4-komşuluk
   ızgarasında BFS tam olarak L1 mesafe dönüşümüdür ve L1 ayrılabilirdir → dört
   `np.minimum.accumulate` taraması. **34× hızlanma** (9,20 → 0,27 ms), çıktı birebir aynı.
2. `_label_components` vektörleştirildi. **Karar hedef cihazda verildi:** geliştirme
   Mac'inde BFS 1,4× kazanıyordu, Jetson'da propagation 1,59× kazandı (3,154 → 1,989 ms;
   kare başına 11 çağrı, yani 34,7 → 21,9 ms). ARM CPU'da Python yorumlayıcısı numpy'a
   göre çok daha yavaş.

**Sonuç:** kare 151,67 → 80,77 ms (**1,88×**), enerji 1,084 → 0,665 J/kare (**1,63×**),
motor değişmedi, çıktılar birebir aynı — diferansiyel testlerle kanıtlı. Kuantalama yok,
doğruluk kaybı yok.

> *Uç cihaz optimizasyon kararları geliştirme makinesinde alınamaz: aynı iki
> implementasyon, aynı girdilerde, iki makinede zıt sonuç verdi.*

### SegFormer anomalisi

SegFormer-B0'ın motoru DDRNet'ten yalnızca **+11,23 ms** yavaş, ama karesi **+193,39 ms**
yavaş — motor farkının **17 katı**. Sebep `output_shape`: SegFormer stride-4'te
(128×256 = 32.768 piksel) logit üretiyor, diğerleri stride-8'de (64×128 = 8.192).
**4× daha fazla piksel** CPU tarafındaki her aşamaya giriyor.

> **Uç cihazda bir modelin sistem maliyetini belirleyen şey FLOP'ları veya GPU gecikmesi
> değil, segmentasyon başının çıktı stride'ıdır** — çünkü CPU tarafındaki algı yığını
> çıktı piksel sayısıyla ölçeklenir, hızlandırıcı ölçeklenmez.

**Bu maliyet giderilebilir bir entegrasyon artığı değil; ölçüldü.** Logitleri stride-8'e
indirmek post-processing'i **3,98×** hızlandırıyor ama **2,03 mIoU'ya mal oluyor**
(0,6721 → 0,6518, 60 Cityscapes val karesi, `scripts/measure_logit_stride_tradeoff.py`).
SegFormer-B0 bu durumda doğruluk sıralamasında birincilikten dördüncülüğe düşer.

Yani stride-4 çıktısı gerçek doğruluk taşıyor: yüksek çözünürlüklü logit hem daha iyi
segmentasyon hem daha yüksek CPU maliyeti demek. Uç cihaz için doğru soru "bu maliyet
kaldırılabilir mi" değil, **"bu doğruluk bu enerjiye değer mi"**.

---

## 9 · Üç eksenin birleşimi

| model | mIoU (yayın) | **mIoU (ölçülen)** | OOD AUROC | OOD AP | yanlış-sürül. | J/kare | FPS |
|---|---|---|---|---|---|---|---|
| PIDNet-M | **80,22** | 0,6847 | 0,562 | 0,124 | 0,0065 | 0,808 | 10,82 |
| PIDNet-S | 78,74 | 0,6768 | 0,667 | 0,156 | 0,0076 | 0,665 | 11,87 |
| DDRNet-23-slim | 77,84 | 0,6850 | 0,674 | 0,165 | **0,0064** | **0,630** | **12,36** |
| SegFormer-B0 | 76,54 | **0,6934** | **0,785** | **0,347** | 0,0091 | 2,221 | 3,57 |
| BiSeNetV2 | 75,76 | 0,6602 | 0,708 | 0,175 | 0,0097 | 0,809 | 10,34 |

### Önce bir düzeltme: hangi mIoU?

Bu tablonun daha önceki hâli tek bir mIoU sütunu taşıyordu ve o sütun **yayınlanmış**
değerlerdi. O sütunla hesaplanan korelasyon çarpıcıydı — ρ(mIoU, AUROC) = −0,90, yani
"doğruluk arttıkça açık küme güvenliği düşüyor". **Bu bulgu geri çekilmiştir.**
Kendi dağıtım koşulumuzda ölçülen mIoU ile aynı hesap:

| ilişki | Spearman ρ | okuma |
|---|---|---|
| yayınlanmış mIoU ↔ AUROC | −0,90 | önceki (geri çekilen) başlık |
| **ölçülen mIoU ↔ AUROC** | **+0,30** | ilişki yok |
| **ölçülen mIoU ↔ AP** | **+0,30** | ilişki yok |
| **yayınlanmış mIoU ↔ ölçülen mIoU** | **+0,10** | *asıl bulgu* |

Yani −0,90'ı üreten şey mimarilerin bir özelliği değil, **yayınlanmış sıralamanın bizim
dağıtım koşulumuzda geçerli olmamasıydı.** ρ(yayın, ölçüm) = +0,10 — model zoo sıralaması
ile 512×1024 dağıtım çözünürlüğünde ölçtüğümüz sıralama arasında pratikte hiçbir ilişki
yok. En doğru yayınlanan model (PIDNet-M, 80,22) ölçümde üçüncü; en düşük yayınlanan
(BiSeNetV2, 75,76) ölçümde de sonuncu ama SegFormer-B0 76,54'ten **birinciliğe** çıkıyor.

> **Tezde kullanılacak cümle bu:** yayınlanmış sıralamalar dağıtım koşuluna aktarılamaz,
> ve aktarılabilir sanmak sahte bir "doğruluk–güvenlik ödünleşimi" üretir. Bunu bir hata
> olarak yaşadık ve düzelttik; §9'un önceki hâli o hatanın kendisidir.

### 9a · "En doğru model" iddiası testi geçmiyor — ve geçmemesi daha ilginç

Yukarıdaki tablo beş modeli sıralıyor ve SegFormer-B0'ı doğrulukta birinci gösteriyor
(0,6934'e karşı 0,6850). Bu sıralama bugüne kadar **hiç test edilmemişti**: toplu kayıtlar
yalnızca veri kümesi düzeyindeki sayıyı tutuyordu, dolayısıyla test edilebilecek bir şey
yoktu. `evaluation/statistics.py` bu iş için `paired_comparison` ve
`deterministic_bootstrap_interval`'ı baştan beri taşıyordu — tam, testli, çağrısız.

Eksik olan kare-başına skorlardı. Beş model **aynı** 500 Cityscapes val karesinde
puanlandı, farklar eşleştirilmiş bootstrap ile ölçüldü
(`scripts/compare_models_paired.py`, `results/paired_comparison.json`):

| karşılaştırma | ortalama fark | %95 aralık | sonuç |
|---|---|---|---|
| **SegFormer-B0 − DDRNet-23-slim** | **+0,0000** | **[−0,0050, +0,0049]** | **ayırt edilemez** |
| PIDNet-M − BiSeNetV2 | +0,0050 | [−0,0001, +0,0108] | ayırt edilemez |
| SegFormer-B0 − PIDNet-M | +0,0063 | [+0,0012, +0,0120] | ayrışıyor |
| PIDNet-M − DDRNet-23-slim | −0,0063 | [−0,0112, −0,0012] | ayrışıyor |
| SegFormer-B0 − PIDNet-S | +0,0195 | [+0,0147, +0,0244] | ayrışıyor |
| PIDNet-S − DDRNet-23-slim | −0,0195 | [−0,0240, −0,0149] | ayrışıyor |

Kare-başına ortalama mIoU: SegFormer-B0 **0,5468**, DDRNet-23-slim **0,5468**. Aynı sayı.

> **Birinci diye bir şey yok.** Doğrulukta bir kazanan değil, **ayırt edilemez bir
> tepe grubu** var. Sekiz çiftin sekizi ayrışıyor, ama ilk ikisi ayrışmıyor — ve tez şu
> ana kadar o ikisi arasındaki 0,0085'lik farkı sıralama diye sunuyordu.

**İki istatistik neden farklı söylüyor?** Veri kümesi mIoU'su 19 sınıfın ortalamasıdır ve
her sınıfa, kaç karede göründüğünden bağımsız olarak eşit ağırlık verir. Kare-başına mIoU
ise her kareye eşit ağırlık verir. Sınıf kırılımı farkı açıklıyor:

| SegFormer lehine | fark | | DDRNet lehine | fark |
|---|---|---|---|---|
| pole | +0,0574 | | bus | −0,0577 |
| terrain | +0,0422 | | train | −0,0448 |
| traffic light | +0,0354 | | fence | −0,0239 |
| person | +0,0325 | | motorcycle | −0,0227 |
| traffic sign | +0,0317 | | rider | −0,0128 |

SegFormer 19 sınıfın **14'ünü** kazanıyor ve kazandıkları ağırlıklı olarak **ince
yapılar**: direk, trafik ışığı, trafik levhası, insan. DDRNet'in kazandığı 5 sınıf ise
ağırlıklı olarak **büyük araçlar**: otobüs, tren. Sınıf-eşit ortalama, 14 küçük tutarlı
kazancı toplayıp **+0,0085** veriyor (kayıtlardan 0,008491; iki mimarinin veri kümesi
mIoU farkı ile sınıf farklarının ortalaması birbirinin aynısıdır, çünkü mIoU zaten
sınıf ortalamasıdır); kare-eşit ortalama ise ikisini başabaş buluyor.

> **Stride-4 hikâyesinin üçüncü bağımsız kanıtı.** SegFormer'ın kazandığı sınıflar —
> direk, trafik ışığı, levha, insan — ince ve uzun yapılardır; kaba ızgarada en çok kaybı
> onlar verir. Aynı mimari tercih §3a'da sınır F1'inde (%36–48 önde), §8'de post-processing
> maliyetinde (3,98×) ve burada sınıf kırılımında görünüyor. Üç farklı ölçüm, tek mekanizma.

**Tezde kurulacak cümle:** *"SegFormer-B0 en doğru modeldir"* değil, *"ilk üç model
kare düzeyinde ayırt edilemez; SegFormer-B0'ın sınıf-ortalamalı üstünlüğü ince yapılardan
gelir ve kare başına 3,53 kat enerjiye mal olur."* İkincisi hem doğru hem daha güçlü:
seçimi doğruluğa değil, enerji ve dayanıklılığa bırakıyor.

**Sınır:** bunlar aynı yayınlanmış checkpoint'lerdir, tek koşudur; bootstrap örnekleme
belirsizliğini ölçer, eğitim tohumu belirsizliğini değil. `significance_claim: false` —
kayıt hiçbir yerde "istatistiksel olarak anlamlı" demiyor, aralığın sıfırı içerip
içermediğini söylüyor.

---

### Kalan gerçek ödünleşim

Doğruluk ile açık küme arasında ilişki yok, ama **enerji** ile açık küme arasında var:
SegFormer-B0 hem ölçülen mIoU'da hem OOD AP'de birinci (0,347, ikincinin 2,1 katı), ve
kare başına **3,53 kat** enerji harcıyor (2,221 J'ye karşı 0,630). Ödünleşim
doğruluk–güvenlik değil, **güvenlik–enerji**.

Mekanizma gözlemi: SegFormer-B0 bu beşlideki tek transformer ve iki bağımsız
"tanıdık-olmayan girdi" testinin ikisinde de önde — bilinmeyen nesneler (AP 0,347'ye
karşı 0,124) ve bilinmeyen koşullar (gece dayanıklılığı %29,7'ye karşı %10,4). n = 5'te
bir gözlemdir, gösterilmiş bir mekanizma değil.

**Karar:** DDRNet-23-slim pratik kazanan — en hızlı, en düşük enerjili, sürülebilir alanda
en güvenli (yanlış-sürülebilir 0,0064) ve ölçülen mIoU'da ikinci. SegFormer-B0 açık küme
ve sınır keskinliğinde açık ara önde ama 3,53× enerjiye mal oluyor **ve bu maliyet
giderilemez**: §8'de ölçüldü, logit'i stride 8'e indirmek post-processing'i 3,98×
hızlandırırken 2,03 mIoU'ya mal oluyor. (§9'un önceki hâli burada "stride düzeltmesi
uygulanırsa maliyet kaybolur" diyordu — bu §8'in ölçümüyle çelişiyordu ve düzeltildi.)

**Dürüstlük sınırı:** n = 5 mimari; bu ölçekte hiçbir ρ kesin kanıt değildir — nitekim
−0,90 tam da bu yüzden yanlış okundu. Ayrıca bunlar farklı reçetelerle (farklı
iterasyon/batch) eğitilmiş yayınlanmış checkpoint'lerdir; kontrollü ablasyon değil, model
karşılaştırmasıdır.

---

## 10 · Ölçülmeyenler

Hiçbiri "ölçülmüş" gibi sunulmayacak.

| eksik | neden | gereken |
|---|---|---|
| ECE / reliability diyagramı | etiketli validasyon verisi yok | Cityscapes val + `evaluate.py run --fit-temperature` |
| Referans modellerin sınıf bazlı IoU'su | aynı | Cityscapes val |
| Gerçek olumsuz-koşul verisi | ACDC indirilmedi | ACDC (kayıt gerektirir) |
| Sürülebilir alan sayısal metriği | `drivable_metrics` çağrısız | GT yol maskeleri |
| SegFormer stride düzeltmesi | zaman | ~20 dk Jetson ölçümü |
| Mühürlü final test verisi | **kasıtlı** | `docs/adr/0005` — yalnızca insan tetikler |

---

## 11 · Üretilen kod

| dosya | ne yapar |
|---|---|
| `scripts/evaluate_open_set.py` | piksel-OOD değerlendirmesi (§5) |
| `scripts/analyze_contextual_risk.py` | yedi özellikli risk füzyonu (§3) |
| `scripts/measure_shift_response.py` | hava/ışık tepkisi (§6) |
| `scripts/jetson/profile_pipeline.py` | aşama profili (§8) |
| `scripts/jetson/compare_labelling.py` | hedef cihazda implementasyon kıyası (§8) |
| `scripts/jetson/run_all_models.sh` | beş modelin gözetimsiz motor+benchmark koşusu (§7) |
| `scripts/build_presentation_outputs.py` | figür/tablo üretimi |

Yol boyunca bulunup düzeltilen gerçek hatalar: `Runner.from_cfg()` ağırlık yüklememesi ·
TF32/FP32 parite uyuşmazlığı · HPO'nun PRUNED denemeleri tamamlanmış sayması · 1980
öncesi mtime'ın log paketini sessizce düşürmesi · `np.trapezoid`'ın numpy 1.26'da
bulunmaması · `threshold_policies`'in O(n²) olması · Jetson benchmark'ının telemetriyi
10 dakikalık ölçümden *sonra* kontrol etmesi.
