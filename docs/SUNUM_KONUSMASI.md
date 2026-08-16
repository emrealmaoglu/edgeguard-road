# Sunum konuşması

**Hedef 5 dakika.** Kayıt, canlı savunma değil — takılırsan durur, o cümleyi tekrar
alırsın.

Format basit: **sayfaya geç, oku.** İmleçle bir şey göstermene gerek yok; ekranda zaten
yazıyor, sen anlat. Ekran kaydı olduğu için hoca isterse durdurup kendisi okur.

- `> "..."` → **söyleyeceğin cümle**, kelimesi kelimesine okuyabilirsin
- **[DUR]** → bir saniye sus, nefes al
- **kalın** → o kelimenin üstüne bas

**Ölçülen süre:** 715 kelime ≈ **5 dakika 5 saniye** konuşma. Duraklamalar, sayfa
geçişleri ve görselleri sessiz gösterdiğin süreyle **5 dakika 50 saniye.**

Tam 5 dakika istiyorsan Sayfa 5'in ilk paragrafını atla ve doğrudan görsellere geç —
**5:30'a** iner. Daha fazlasını kesmeni önermem; kalanların hepsi bir bulguyu taşıyor.

---

## Sayfa 1 · Problem *(0:00 – 0:45)*

> "Bir otoyolda gidiyorsunuz. Önünüzdeki kamyondan bir kasa düşüyor.
>
> **[DUR]**
>
> İnsan sürücü ne olduğunu anlamaz — ama bilmediği bir şey olduğunu anlar ve yavaşlar.
>
> Peki bir segmentasyon modeli ne yapar?
>
> **[DUR]**
>
> Onu **yol** diye sınıflandırır, ve bundan yüzde seksen beş **emin** olur. Çünkü
> öğrendiği on dokuz sınıfın arasında 'düşmüş kasa' yok — en yakın gördüğü şeye koyuyor.
>
> Sorun modelin yanılması değil. Sorun, **yanıldığını bilmemesi.**
>
> Bu proje tam olarak bunu ölçüyor: doğruluk, **uncertainty**, **open-set** ve uç cihaz
> maliyeti. Dört eksen."

*Aşağıdaki sistem mimarisi şemasını göster, birkaç saniye sessiz bekle.*

> "Sistemin akışı bu: kare, TensorRT motoru, sonra CPU tarafında maske, güven haritası
> ve risk sıralaması. Panel de bu cihazın **üzerinde** çalışıyor — sol üstteki watt ve
> sıcaklık Jetson'ın şu anki gerçek değerleri."

---

## Sayfa 2 · Model karşılaştırması *(0:45 – 1:35)*

> "Beş tane real-time segmentasyon mimarisi aldık; hepsini aynı protokolle, aynı veriyle,
> kendi deployment çözünürlüğümüzde ölçtük.
>
> Tabloda soldaki sütun literatürün yayınladığı **mIoU**, sağdaki bizim ölçtüğümüz.
>
> **[DUR]**
>
> Sıralama **tutmuyor.** Aralarındaki korelasyon artı sıfır nokta on — pratikte ilişki yok.
> Yayınlanmış sıralamada sonuncu olan SegFormer, bizim ölçümümüzde başa çıkıyor.
>
> Yani uç cihaza model seçerken model zoo'daki sayıya bakamazsınız. Kendiniz ölçmek
> zorundasınız.
>
> **[DUR]**
>
> İkinci bulgu: 'en doğru model hangisi' sorusunun cevabı **yok.** Beş modeli aynı beş yüz
> karede puanlayıp farkın gerçek mi rastlantı mı olduğunu **bootstrap** ile test ettik.
> İlk iki model arasındaki fark sıfır çıktı, güven aralığı sıfırı içeriyor.
>
> Doğrulukta bir kazanan değil, **ayırt edilemez bir tepe grubu** var. Bu da seçimi
> otomatik olarak enerjiye ve dayanıklılığa bırakıyor."

---

## Sayfa 3 · Open-set *(1:35 – 2:15)*

> "Open-set derken modelin **hiç görmediği** şeyleri kastediyoruz. Altmış karelik, piksel
> piksel etiketlenmiş gerçek yol tehlikesi verisi kullandık — hiçbir model bununla
> eğitilmedi.
>
> Üstteki tabloda dört farklı uncertainty skorunu karşılaştırdık. Sıralama literatürün
> söylediğiyle aynı çıktı: **energy** en iyisi, **softmax** en kötüsü.
>
> **[DUR]**
>
> Asıl bulgu alttaki tabloda. Tehlikeleri türüne göre ayırdığımızda, yola düşmüş yükte
> **AUROC sıfır nokta kırk sekiz.**
>
> Sıfır nokta elli yazı-tura demek. Yani sistem, yolda duran bir kargoyu yazı-turadan
> **daha kötü** ayırt ediyor. Açılışta anlattığım kasa — ölçtük, gerçekten fark etmiyor.
>
> Bu sayıyı gizlemedik, çünkü güvenlik açısından en önemli sayı bu."

---

## Sayfa 4 · Kalibrasyon *(2:15 – 2:50)*

> "**Calibration** şu demek: model 'yüzde doksan eminim' dediğinde gerçekten yüzde doksan
> haklı mı? Üstteki tabloda görüldüğü gibi, model **kesinlikle yanıldığı** piksellerde
> bile yüzde yetmiş üç ile seksen dört arası güven veriyor.
>
> **[DUR]**
>
> Alttaki tablo daha önemli. Tek bir **ECE** sayısı yanıltıyor: sahnenin yüzde otuz
> dokuzu asfalt, model orada hem çok emin hem haklı, ve o kütle küçük sınıflardaki aşırı
> güveni **yutuyor.** Class-wise hesapladığımızda hata **bir buçuk ile üç kat** büyüyor —
> her modelde en kötü sınıf aynı: direk ve çit."

---

## Sayfa 5 · Bağlamsal risk ve görseller *(2:50 – 3:35)*

> "Bulunan bölgeler yedi özellikli, **açıklanabilir** bir füzyonla sıralanıyor — kara kutu
> bir skor değil. Alttaki sürülebilir alan tablosunda önemli olan sütun
> **yanlış-sürülebilir**: aracın gireceği ama yol olmayan piksellerin oranı."

*Aşağı kaydır, görselleri sırayla göster. Her birinde 3-4 saniye sessiz bekle.*

> "Şurada aynı karede beş mimarinin çıktısı var.
>
> Bu segmentasyon, bu sürülebilir koridor, bu bulunan bölgeler, bu da operasyonel dikkat
> haritası — sistemin hangi bölgeye öncelik verdiği."

---

## Sayfa 6 · Uç cihaz *(3:35 – 4:25)*

> "Uç ölçümleri gerçek cihazda alındı: yirmi beş watt modunda, altı yüz saniye kesintisiz
> yük altında.
>
> Bu eğriler o koşu sırasında kaydedildi. Sıcaklığın düz gitmesi **thermal throttling**
> olmadığını gösteriyor — yani bu sayılar cihazın ısınmadan önceki iyi anları değil,
> sürdürebildiği gerçek performans."

*Aşağı kaydır, benchmark tablosuna gel.*

> "Ve projenin mühendislik açısından en önemli bulgusu burada.
>
> SegFormer'ın TensorRT motoru DDRNet'ten sadece **on bir milisaniye** yavaş. Ama uçtan uca
> karesi **yüz doksan üç milisaniye** yavaş.
>
> **[DUR]**
>
> Motor farkının **on yedi katı.**
>
> Sebebi output **stride**: SegFormer çıktısını daha ince ızgarada üretiyor, bu da CPU
> tarafına dört kat piksel veriyor. Yani uç cihazda sistem maliyetini FLOP ya da model
> boyutu değil, segmentasyon başının çıktı stride'ı belirliyor.
>
> Motorun kare bütçesindeki payı zaten sadece yüzde beş. O yüzden optimizasyonu modele
> değil koda yaptık: kare süresi yüz kırk altı milisaniyeden doksan dörde indi, modele hiç
> dokunmadan."

*Aşağı kaydır, demo videosunu göster. Oynatıp 5-6 saniye sessiz izlet.*

> "Bu da hareketli sahnede sistemin çıktısı — Cityscapes demo videosu, yüz seksen kare,
> önceden üretildi."

---

## Sayfa 7 · Sınırlar ve kapanış *(4:25 – 5:00)*

> "Sınırlar. Real-time hedefini **geçemedik** — en iyi sonuç saniyede on iki buçuk kare,
> hedef yirmiydi. Ama darboğazın nerede olduğunu ölçtük: model değil, CPU tarafı.
>
> Bir de bu projede **yanlış çıkan iki sonuç** oldu, ikisi de burada duruyor. Bir ara
> 'doğruluk arttıkça open-set güvenliği düşüyor' diye çarpıcı bir korelasyon bulmuştuk;
> **geri çektik.** Çünkü o korelasyon yayınlanmış mIoU ile hesaplanmıştı, kendi
> ölçtüğümüzle ilişki sıfır çıktı.
>
> **[DUR]**
>
> Toparlayayım. Uç cihaz için model seçimi yayınlanmış doğruluğa bakarak yapılamaz;
> doğrulukta ilk grup ayrışmadığı için seçim enerjiye kalıyor; ve sistem maliyetini output
> stride belirliyor.
>
> Başta bir kasa düşmüştü. Ölçtük — sistem onu fark etmiyor. Bu proje o sorunu **çözmüyor**,
> ama nerede olduğunu ve neden olduğunu ölçüyor.
>
> Panelde gördüğünüz her sayı diskteki bir ölçüm kaydından okunuyor; hiçbiri elle yazılmadı.
>
> Teşekkür ederim."

---

## Kayıt notları

**Çekimden önce:** tarayıcı tam ekran (F11) · sayfa 1'de bekliyor · telefon sessiz ·
bildirimler kapalı · mikrofonu bir cümleyle test et.

**Konuşurken:**

- Sayfa geçtikten sonra **bir saniye sus**, sonra konuş. En sık atlanan şey bu.
- **Tabloları okuma.** Sayfa başına bir-iki sayıyı sesli söyle, gerisi ekranda dursun.
- Görsellerde **sessiz bekle.** Konuşurken geçiştirme — video ve resimler kendi başlarına
  anlatıyor.
- Kayıtta herkes hızlanır. Bilerek yavaş konuş.

**Ters giderse:**

| durum | ne yap |
|---|---|
| Cümleyi bozdun | Dur, üç saniye bekle, cümleyi baştan al — sonra o kısmı kesersin |
| Panel takıldı | F5, kaldığın sayfaya dön, o bölümü tekrar al |
| Süre aşıyor | Sayfa 5'in ilk paragrafını kes, doğrudan görsellere geç |
| Süre kısa kaldı | Sayfa 6'daki "Koşu" menüsünden başka model seç, eğriler yenilensin |

**Ezberle:** `+0,10` · `0,48` · `17 kat`. Gerisi ekranda yazıyor.

---

## Sorulursa

*(Sözlü bir kısım olursa hazır dursun.)*

**Kendi modelinizi eğitmediniz mi?**
> Eğittik, Cityscapes ve IDD20K karışımıyla — ama sınırlı bütçede, yayınlanmış
> checkpoint'lerin yüzde 0,7'si kadar örnekle. O yüzden karşılaştırma temeli olarak resmî
> referansları kullandık ve ikisini karıştırmadık.

**n = 5 ile korelasyon konuşulur mu?**
> Konuşulmaz, tezde de öyle yazıyor — nitekim o eksi 0,90 bulgusunu bu yüzden geri çektik.
> Sayısal iddialarımız korelasyondan değil, aynı karelerde yapılan eşleştirilmiş
> karşılaştırmadan geliyor.

**FP16'ya çevirince doğruluk kaybettiniz mi?**
> Ölçtük. Cihazda FP16 motorla FP32'yi aynı koşuda aynı karelerde puanladık: fark sıfırdan
> ayırt edilemiyor, piksel uyumu yüzde 99,97.

**12 FPS real-time sayılır mı?**
> Sayılmaz, öyle de demiyoruz — sınırlar sayfasında "geçilmedi" yazıyor. Ölçtüğümüz şey
> darboğazın nerede olduğu.

**Bu bir ürün mü?**
> Hayır, araştırma prototipi. ISO 26262 kapsamında bir güvenlik seviyesi hedeflenmedi.
> Bulgular ISO 21448 — SOTIF — sınıfında: sistem arızalanmadan, sadece algı yetersiz
> kaldığı için ortaya çıkan tehlikeler.
