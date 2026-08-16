# Sunum konuşması — 6 dakika, panelle senkron

**Kurulum:** panel Jetson'da açık, tarayıcı tam ekran, 1. sayfada bekliyor. Kenar
çubuğundaki canlı watt/sıcaklık görünür durumda — bunu bir kez sözle işaretle, sonra
unut.

**Omurga — sunum boyunca kaybetme:** *"Beş model karşılaştırdık. Asıl bulgu modellerde
değil, karşılaştırmanın kendisinde çıktı."*

**Uzunluk:** metin ~776 kelime. Net ve yavaş konuşurken (~140 kelime/dk) **5,5 dakika**,
normal tempoda 4,8. Yani sayfa geçişleri, nefes ve bir-iki duraklama için **30-60 saniye
payın var** — acele etme, bu tampon bilerek bırakıldı.

Zaman işaretleri hedeftir, kanun değil. Sarkıyorsan **§5'i (bağlamsal risk) atla** —
tek gerçekten sıkıştırılabilir bölüm odur.

---

## 0:00 – 0:40 · Problem (Sayfa 1)

> "Otonom sürüş algı sistemlerinin sessiz bir hatası var: eğitimde görmedikleri bir şeyle
> karşılaştıklarında yanlış cevap vermiyorlar — **yanlış cevaptan emin oluyorlar.**
>
> Bu tezin sorusu şu: kaynak kısıtlı bir uç cihazda, hem gördüğünü söyleyen hem de ne
> kadar emin olduğunu söyleyen bir sistem kurulabilir mi? Hem de 25 watt bütçe içinde.
>
> Şu anda gördüğünüz panel, anlattığı cihazın **üzerinde** çalışıyor. Sol üstteki güç ve
> sıcaklık, Jetson'ın şu andaki gerçek değerleri."

**Ekranda:** dört eksen kutusunu göster (doğruluk · belirsizlik · açık küme · uç maliyet),
sonra aşağıdaki sistem mimarisi şemasını 3-4 saniye göster.

> "Dört eksende ölçtük. Sırayla gidelim."

---

## 0:40 – 1:40 · İlk bulgu: yayınlanmış sıralama taşınmıyor (Sayfa 2)

**Sayfa 2'ye geç.** Tabloyu göster, **"Yayın mIoU" ve "Ölçülen mIoU" sütunlarını
parmakla/imleçle işaretle.**

> "Beş gerçek zamanlı mimariyi aynı protokolle, kendi dağıtım çözünürlüğümüzde ölçtük.
>
> İlk bulgu burada. Soldaki sütun literatürün yayınladığı mIoU, sağdaki bizim ölçtüğümüz.
> **Sıralama tutmuyor.** Spearman korelasyonu artı **sıfır nokta on** — yani pratikte
> hiçbir ilişki yok.
>
> Yayınlanmış sıralamada **sonuncu** olan SegFormer, bizim dağıtım koşulumuzda başa
> çıkıyor. Bunun pratik anlamı şu: **uç cihaz için model seçimi yayınlanmış doğruluğa
> bakılarak yapılamaz.** Kendiniz ölçmek zorundasınız."

---

## 1:40 – 2:20 · İkinci bulgu: doğrulukta kazanan yok (Sayfa 2, üst kutular)

**Yukarı kaydır, "Doğrulukta / berabere" kutusunu göster.**

> "İkinci bulgu daha da ilginç. 'En doğru model hangisi' sorusunun cevabı yok.
>
> Beş modeli **aynı 500 karede** puanlayıp farkı eşleştirilmiş bootstrap ile test ettik.
> İlk iki model arasındaki fark **artı sıfır nokta sıfır sıfır sıfır sıfır** — güven
> aralığı sıfırı içeriyor. On çiftin sekizi ayrışıyor; **ilk ikisi ayrışmıyor.**
>
> Yani doğrulukta bir kazanan değil, **ayırt edilemez bir tepe grubu** var. Bu, seçimi
> otomatik olarak diğer eksenlere bırakıyor: enerji ve dayanıklılık."

---

## 2:20 – 3:00 · Açık küme: güvenlik bulgusu (Sayfa 3)

**Sayfa 3'e geç.** Skor tablosunu kısa göster, sonra **tehlike türü tablosuna in ve
`lost` satırını işaretle.**

> "Açık kümeyi 60 karelik, piksel etiketli gerçek yol tehlikesi üzerinde ölçtük — hiçbir
> model bu veriyle eğitilmedi.
>
> Skor sıralaması literatürle aynı çıktı: energy en iyisi, softmax en kötüsü.
>
> Ama asıl bulgu şurada — tehlike türüne göre kırdığımızda. **Yola düşmüş yükte AUROC
> sıfır nokta kırk sekiz.** Bu **yazı-turadan kötü** demek. Sistem, yolda duran bir kargoyu
> fark edemiyor. Bunu gizlemiyoruz, çünkü güvenlik açısından en önemli sayı bu."

---

## 3:00 – 3:40 · Kalibrasyon: havuzlanmış sayı yanıltıyor (Sayfa 4)

**Sayfa 4'e geç, "Havuzlanmış ECE neyi saklıyor" tablosuna in.**

> "Kalibrasyon, modelin güveninin gerçeği yansıtıp yansıtmadığı. Standart ölçüsü ECE.
>
> Ama tek bir ECE yanıltıcı. Bir sürüş sahnesinin **yüzde otuz dokuzu asfalt** ve model
> asfaltta hem çok emin hem haklı. O kütle, önemli sınıflardaki aşırı güveni **soğuruyor.**
>
> Sınıf bazlı hesapladığımızda hata **bir buçuk ile üç kat** büyüyor. Ve her modelde en
> kötü sınıf aynı: **direk ve çit.** PIDNet-S direkte kendi havuzlanmış değerinin **sekiz
> katı** hata veriyor — üstelik emin olarak."

---

## 3:40 – 4:00 · Bağlamsal risk (Sayfa 5) — *sıkışırsan atla*

**Sayfa 5'e geç, risk tablosunu ve altındaki zamansal bloğu göster.**

> "Bulunan bölgeler yedi özellikli açıklanabilir bir füzyonla sıralanıyor. Ölçülemeyen
> özellikler sıfır **değerle** değil sıfır **ağırlıkla** dışlanıyor — sıfır değer,
> ölçülmemiş bir sinyali 'risk yok' gibi gösterirdi.
>
> Zamansal ölçüm şunu söylüyor: sistemin işaretlediğinin **yarısı tek kare yaşıyor.**
> Yani tek-kare dikkat, istisna olarak değil baskın davranış olarak titriyor."

---

## 4:00 – 5:00 · Uç maliyet ve mekanizma (Sayfa 6)

**Sayfa 6'ya geç.** Önce telemetri eğrilerini göster, sonra benchmark tablosuna in.

> "Uç ölçümleri gerçek cihazda, 25 watt modunda, **600 saniye sürdürülen yük** altında
> alındı. Eğrilerin düzleşmesi termal kısma olmadığını gösteriyor.
>
> Ve burada projenin mühendislik açısından en önemli bulgusu var."

**"Motor" ve "Kare" sütunlarını yan yana işaretle.**

> "SegFormer'ın TensorRT motoru DDRNet'ten sadece **11 milisaniye** yavaş. Ama uçtan uca
> karesi **193 milisaniye** yavaş — motor farkının **17 katı.**
>
> Sebep: SegFormer çıktısını daha ince ızgarada üretiyor, bu da CPU tarafına **dört kat
> piksel** veriyor. Yani **uç cihazda sistem maliyetini FLOP veya model boyutu değil,
> segmentasyon başının çıktı stride'ı belirliyor.**
>
> Ve bu tek parametre, sunum boyunca saydığım bütün başarısızlıkları açıklıyor: ince
> yapılar — direk, ışık, levha, insan — hem kötü segmentleniyor, hem kötü kalibre
> ediliyor, hem parçalanıyor. **Beş ayrı ölçüm, tek mekanizma.**"

**Aşağı kaydır, kare bütçesi tablosunu göster.**

> "Motorun kare bütçesindeki payı yüzde beş. Optimizasyonu modele değil koda yaptık: kare
> **146 milisaniyeden 94'e** indi, modele hiç dokunmadan, çıktılar birebir aynı kalarak."

---

## 5:00 – 5:40 · Sınırlar ve geri çekmeler (Sayfa 7)

**Sayfa 7'ye geç.** Bu bölümü **savunarak değil, sahiplenerek** anlat.

> "Sınırlar. Gerçek zaman hedefini **geçemedik** — en iyi 12,4 FPS, hedef 20. Darboğazın
> model değil CPU tarafı olduğunu ölçtük, yani yol açık ama yürünmedi.
>
> Bu projede **yanlış çıkan iki sonuç** var ve ikisi de burada duruyor."

**"Geri çekilen iki iddia" bölümünü göster.**

> "Bir ara 'doğruluk arttıkça açık küme güvenliği düşüyor' diye güçlü bir korelasyon
> bulmuştuk — eksi sıfır doksan. **Geri çektik.** O korelasyon yayınlanmış mIoU ile
> hesaplanmıştı; kendi ölçtüğümüzle ilişki sıfır. Bulunan şey mimarilerin özelliği değil,
> yayınlanmış sıralamanın taşınmamasıydı.
>
> İkincisi de ölçümle düzeltildi. Ölçüm yapan bir çalışmanın en zayıf yeri, düzeltmediği
> hatalardır."

---

## 5:40 – 6:00 · Kapanış

> "Özetle: uç cihaz için model seçimi yayınlanmış doğruluğa göre yapılamaz; doğrulukta
> ilk grup ayrışmadığı için seçim enerji ve dayanıklılığa kalıyor; ve sistem maliyetini
> çıktı stride'ı belirliyor.
>
> Panelde gördüğünüz **her sayı diskteki bir ölçüm kaydından** okunuyor — panel canlı
> çıkarım yapmıyor, hiçbir sayı elle yazılmadı. Teşekkürler."

---

## Jüri sorabilir — hazır cevaplar

**"Neden kendi modelinizi eğitmediniz?"**
> Eğittik — Cityscapes+IDD20K karışımıyla, sınırlı bütçede. Ama o modeller yayınların
> yüzde 0,7'si kadar örnek gördü, o yüzden karşılaştırma temeli olarak resmî referans
> checkpoint'leri kullandık ve ikisini **ayrı** tuttuk. Tezde ikisi de var, karıştırılmadı.

**"n=5 ile korelasyon konuşulur mu?"**
> Konuşulmaz, ve tezde de öyle yazıyor. Nitekim eksi 0,90 bulgusunu tam bu yüzden geri
> çektik. Sayısal iddialarımız korelasyondan değil, **eşleştirilmiş karşılaştırmadan**
> geliyor — aynı karelerde, güven aralığıyla.

**"FP16'ya çevirince doğruluk kaybetmediniz mi?"**
> Ölçtük. Cihazda, FP16 motoru ile FP32'yi **aynı koşuda aynı karelerde** puanladık: fark
> sıfırdan ayırt edilemiyor, piksel uyumu %99,97. Eşleştirmeseydik yanlış bir sonuç
> bulacaktık — ayrı ölçülmüş sayılarla kıyaslayınca DDRNet'te 0,025'lik sahte bir ceza
> çıkıyordu, ki tamamı kare alt kümesinden geliyordu.

**"12 FPS ile gerçek zamanlı denebilir mi?"**
> Denemez, ve demiyoruz. Sınırlar sayfasında **geçilmedi** yazıyor. Ölçtüğümüz şey
> darboğazın nerede olduğu: motorun payı %5, geri kalanı CPU tarafı.

**"Bu bir ürün mü?"**
> Hayır, araştırma prototipi. ISO 26262 kapsamında hiçbir güvenlik seviyesi hedeflenmedi
> ve tezde bu açıkça yazılı. Ama bulgular **ISO 21448 / SOTIF** sınıfında: sistem
> arızalanmadan, sadece algı yetersiz kaldığı için ortaya çıkan tehlikeler.

---

## Prova notları

- **En sık yapılan hata:** her sayfada durup her tabloyu okumak. Tablolar arka plan,
  konuşma ön planda. Sayfa başına **bir** sayıyı sesli söyle, gerisi ekranda dursun.
- Sayfa geçişlerinde 1 saniye bekle — izleyicinin gözü yeni sayfaya otursun.
- **Üç sayıyı ezberle**, kâğıda bakma: **+0,10** (sıralama taşınmıyor), **0,4805** (kayıp
  yük), **17 kat** (motor–kare uçurumu).
- Prova sırasında süreyi tut. 6 dakikayı aşarsan **önce §5'i**, sonra §3'ün skor
  tablosunu kes.
