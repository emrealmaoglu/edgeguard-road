# Sunum konuşması — yönetmen metni

**Bu bir kayıt, canlı savunma değil.** Takılırsan durur, o cümleyi tekrar alırsın. Kimse
araya girmeyecek. Ton buna göre: **anlatıyorsun, savunmuyorsun.**

## Nasıl okunur

| işaret | anlamı |
|---|---|
| `> "..."` | **Söyleyeceğin cümle.** Kelimesi kelimesine okuyabilirsin. |
| **[DUR]** | Bir saniye sus. Nefes al. Bu boşluk kayıtta çok iyi durur. |
| **kalın** | Sesini hafif yükselt, o kelimenin üstüne bas. |
| `▸ Ekran:` | O anda ekranda ne olacak. |
| `▸ Yap:` | Fareyle/klavyeyle ne yapacaksın. |

**Omurga — sunum boyunca kaybetme:**
*"Beş model karşılaştırdık. Asıl bulgu modellerde değil, karşılaştırmanın kendisinde çıktı."*

**Uzunluk — ölçüldü, tahmin değil:** 888 kelime + 15 duraklama + 8 sayfa geçişi.
Dakikada 140 kelimeyle **6 dakika 50 saniye.** Üst sınır 7 dakika olduğu için **payın
dar** — iki sonucu var:

- **Hızlanma.** Kayıtta herkes hızlanır, sen de hızlanırsan 6 dakikaya iner ve
  duraklamalar kaybolur. Metnin ritmi onlara göre kuruldu.
- **Sarkarsan §5'i atla** (Bağlamsal risk, 4:00–4:20). Bu **20 saniye** kazandırır ve
  seni 6:30'a indirir. Kararı çekim sırasında değil, **şimdi** ver: saatine bakıp
  5. sayfaya geldiğinde 4 dakikayı geçtiysen atla.

**Ezberle, kâğıda bakma:** `+0,10` · `0,4805` · `17 kat`. Gerisi ekranda zaten yazıyor.

---

# ⏱ 0:00 – 0:55 · Açılış

`▸ Ekran:` Sayfa 1, tarayıcı tam ekran (F11).
`▸ Yap:` Konuşmaya başlamadan önce **iki saniye bekle.** Kayıt otursun.

> "Bir otoyolda gidiyorsunuz. Önünüzdeki kamyondan bir kasa düşüyor.
>
> **[DUR]**
>
> İnsan sürücü ne yapar? Ne olduğunu anlamaz — ama bilmediği bir şey olduğunu anlar. Ve
> yavaşlar.
>
> Peki bir yapay zekâ algı sistemi ne yapar?
>
> **[DUR]**
>
> Onu **yol** diye sınıflandırır. Ve bundan yüzde seksen beş **emin** olur.
>
> Çünkü ona öğretilen on dokuz sınıfın arasında 'düşmüş kasa' diye bir şey yok. Modelin
> elinde o kutuyu koyacağı bir yer yok — en yakın gördüğü şeye koyuyor. Asfalta.
>
> **[DUR]**
>
> Bu tezin çıkış noktası tam olarak bu. Sorun modelin **yanılması** değil. Sorun,
> yanıldığını **bilmemesi.**"

`▸ Yap:` İmleci sol üstteki canlı değerlere götür, bir saniye bekle.

> "Şu anda gördüğünüz panel, anlattığı cihazın **üzerinde** çalışıyor. Sol üstteki güç ve
> sıcaklık, Jetson'ın şu andaki gerçek değerleri — bir yerden alınmış ekran görüntüsü
> değil."

`▸ Yap:` İmleci dört eksen kutusunun üstünde soldan sağa gezdir.

> "Projeyi dört eksende ölçtük: doğruluk, belirsizlik, açık küme ve uç cihaz maliyeti.
> Sırayla gidelim."

---

# ⏱ 0:55 – 1:55 · İlk bulgu (Sayfa 2)

`▸ Yap:` Sol menüden **"2 · Model karşılaştırması"**. Bir saniye bekle.
`▸ Ekran:` Beş satırlık büyük tablo.

> "Beş tane gerçek zamanlı segmentasyon mimarisi aldık. Hepsini **aynı protokolle**, aynı
> veriyle, kendi dağıtım çözünürlüğümüzde ölçtük.
>
> Ve ilk bulgu burada çıktı."

`▸ Yap:` İmleci **"Yayın mIoU"** sütununda yukarıdan aşağı indir, sonra **"Ölçülen mIoU"**
sütununda aynısını yap. Karşılaştırıyor gibi.

> "Soldaki sütun literatürün yayınladığı doğruluk. Sağdaki bizim ölçtüğümüz.
>
> **[DUR]**
>
> Sıralama **tutmuyor.** İki sütun arasındaki korelasyon artı **sıfır nokta on** — yani
> pratikte hiçbir ilişki yok.
>
> Bakın: yayınlanmış sıralamada **en sonda** olan SegFormer, bizim ölçümümüzde **en
> başa** çıkıyor.
>
> **[DUR]**
>
> Bunun pratik anlamı şu: uç cihaza model seçerken model zoo'daki sayıya bakamazsınız.
> **Kendiniz ölçmek zorundasınız.** Bu tezin ilk katkısı bu."

---

# ⏱ 1:55 – 2:40 · İkinci bulgu (Sayfa 2, üst)

`▸ Yap:` Sayfanın başına kaydır. Dört metrik kutusu görünsün.

> "İkincisi daha da ilginç. 'En doğru model hangisi' diye sorduk. Cevabı **yok.**"

`▸ Yap:` İmleci **"Doğrulukta — berabere"** kutusunun üstüne koy, orada bırak.

> "Şöyle test ettik: beş modeli de **aynı beş yüz karede** puanladık, sonra aradaki farkın
> gerçek mi yoksa rastlantı mı olduğunu bootstrap ile ölçtük.
>
> İlk iki model arasındaki fark: **artı sıfır, sıfır, sıfır, sıfır.** Güven aralığı sıfırı
> içeriyor.
>
> **[DUR]**
>
> On model çiftinin sekizi ayrışıyor. **İlk ikisi ayrışmıyor.**
>
> Yani doğrulukta bir kazanan yok — **ayırt edilemez bir tepe grubu** var. Ve bu, seçimi
> otomatik olarak başka eksenlere bırakıyor: enerji ve dayanıklılık."

---

# ⏱ 2:40 – 3:20 · Açık küme (Sayfa 3)

`▸ Yap:` Sol menüden **"3 · Açık küme yol tehlikesi"**.

> "Açık küme derken şunu kastediyoruz: modelin **hiç görmediği** bir şey.
>
> Altmış karelik, piksel piksel etiketlenmiş gerçek yol tehlikesi verisi kullandık.
> Hiçbir model bu veriyle eğitilmedi."

`▸ Yap:` Üstteki skor tablosunu göster, **üç saniye**, sonra aşağı kaydır.

> "Dört farklı belirsizlik skorunu karşılaştırdık. Sıralama literatürün söylediğiyle aynı
> çıktı — bu da uygulamamızın doğru çalıştığının bağımsız bir kanıtı.
>
> Ama asıl bulgu aşağıda."

`▸ Yap:` İmleci **`lost` satırının** üstüne koy ve orada bırak.

> "Tehlikeleri türüne göre ayırdığımızda: yola düşmüş yükte AUROC **sıfır nokta kırk
> sekiz.**
>
> **[DUR]**
>
> Sıfır nokta elli, yazı-tura demek. Yani sistem, yolda duran bir kargoyu **yazı-turadan
> daha kötü** ayırt ediyor.
>
> Açılışta anlattığım kasa — ölçtük, gerçekten fark edemiyor.
>
> Bu sayıyı gizleyebilirdik. Gizlemedik, çünkü güvenlik açısından bu tezin **en önemli
> sayısı** bu."

---

# ⏱ 3:20 – 4:00 · Kalibrasyon (Sayfa 4)

`▸ Yap:` Sol menüden **"4 · Belirsizlik ve kalibrasyon"**, sonra aşağı kaydır —
**"Havuzlanmış ECE neyi saklıyor"** tablosuna kadar.

> "Kalibrasyon şu demek: model 'yüzde doksan eminim' dediğinde, gerçekten yüzde doksan
> haklı mı?
>
> Standart ölçüsü ECE. Ama tek bir ECE sayısı **yanıltıyor** — ve bunu ölçtük."

`▸ Yap:` İmleci **"Havuzlanmış ECE"** ile **"Sınıf-bazlı ECE"** sütunları arasında gezdir.

> "Bir sürüş sahnesinin yüzde otuz dokuzu **asfalt.** Model asfaltta hem çok emin, hem de
> haklı. O büyük kütle, küçük sınıflardaki aşırı güveni **yutuyor.**
>
> Sınıf bazlı hesapladığımızda hata **bir buçuk ile üç kat** büyüyor.
>
> **[DUR]**
>
> Ve her modelde en kötü sınıf aynı çıkıyor: **direk ve çit.** Şuradaki PIDNet, direkte
> kendi havuzlanmış değerinin **sekiz katı** hata veriyor. Üstelik emin olarak."

---

# ⏱ 4:00 – 4:20 · Bağlamsal risk (Sayfa 5) · *sarkarsan bu bölümü atla*

`▸ Yap:` Sol menüden **"5 · Bağlamsal risk"**.

> "Bulunan bölgeler yedi özellikli, **açıklanabilir** bir füzyonla sıralanıyor — kara kutu
> bir skor değil; hangi etkenin ne kadar katkı verdiği yazılı.
>
> Ölçemediğimiz bir özelliği sıfır **değerle** değil, sıfır **ağırlıkla** dışlıyoruz.
> Sıfır değer verseydik, ölçülmemiş bir sinyali 'risk yok' gibi gösterirdi.
>
> Zamansal ölçüm de şunu söylüyor: sistemin işaretlediğinin **yarısı tek kare yaşıyor.**"

---

# ⏱ 4:20 – 5:20 · Uç cihaz ve mekanizma (Sayfa 6)

`▸ Yap:` Sol menüden **"6 · Uç cihaz: koşu telemetrisi"**.
`▸ Ekran:` Üç grafik — güç, RAM, sıcaklık.

> "Uç ölçümlerini gerçek cihazda aldık. Yirmi beş watt modunda, **altı yüz saniye
> kesintisiz yük** altında.
>
> Bu eğriler o koşu sırasında kaydedildi. Sıcaklığın düz gitmesi önemli — **termal kısma
> olmadığını** gösteriyor. Yani bu sayılar cihazın ısınmadan önceki iyi anları değil,
> sürdürebildiği gerçek performans."

`▸ Yap:` Aşağı kaydır, benchmark tablosuna gel.

> "Ve burada projenin mühendislik açısından en önemli bulgusu var."

`▸ Yap:` İmleci önce **"Motor"** sütununa, sonra **"Kare"** sütununa götür. İkisi arasında
bir kez git-gel yap.

> "SegFormer'ın TensorRT motoru, DDRNet'ten sadece **on bir milisaniye** yavaş.
>
> Ama uçtan uca karesi **yüz doksan üç milisaniye** yavaş.
>
> **[DUR]**
>
> Motor farkının **on yedi katı.**
>
> Sebebi şu: SegFormer çıktısını daha ince bir ızgarada üretiyor. Bu da CPU tarafındaki
> işleme **dört kat piksel** veriyor. Hızlandırıcı ölçekleniyor, CPU ölçeklenmiyor.
>
> **[DUR]**
>
> Yani uç cihazda sistem maliyetini FLOP sayısı ya da model boyutu değil, **segmentasyon
> başının çıktı stride'ı** belirliyor.
>
> Ve bu tek parametre, sunum boyunca saydığım bütün başarısızlıkları açıklıyor. İnce
> yapılar — direk, trafik ışığı, levha, insan — hem kötü segmentleniyor, hem kötü kalibre
> ediliyor, hem parçalanıyor. **Beş ayrı ölçüm, tek mekanizma.**"

`▸ Yap:` Aşağı kaydır, kare bütçesi tablosunu göster.

> "Son olarak: motorun kare bütçesindeki payı sadece **yüzde beş.** O yüzden optimizasyonu
> modele değil koda yaptık. Kare süresi **yüz kırk altı milisaniyeden doksan dörde** indi —
> modele hiç dokunmadan, çıktılar birebir aynı kalarak."

---

# ⏱ 5:20 – 6:00 · Sınırlar (Sayfa 7)

`▸ Yap:` Sol menüden **"7 · Sınırlar"**. Bu bölümü **sakin** anlat, acele etme.

> "Sınırlar. Bunları sona sakladım ama gizlemiyorum.
>
> Gerçek zaman hedefini **geçemedik.** En iyi sonuç saniyede on iki buçuk kare, hedef
> yirmiydi. Ama darboğazın nerede olduğunu ölçtük: model değil, CPU tarafı. Yani yol
> belli, sadece yürünmedi.
>
> **[DUR]**
>
> Bir de şu var: bu projede **yanlış çıkan iki sonuç** oldu. İkisi de burada duruyor."

`▸ Yap:` **"Geri çekilen iki iddia"** bölümünü göster.

> "Bir ara çok çarpıcı bir korelasyon bulmuştuk — 'doğruluk arttıkça açık küme güvenliği
> düşüyor.' **Geri çektik.**
>
> Çünkü o korelasyon **yayınlanmış** doğrulukla hesaplanmıştı. Kendi ölçtüğümüz
> değerlerle yaptığımızda ilişki sıfır çıktı. Bulduğumuz şey mimarilerin bir özelliği
> değil, yayınlanmış sıralamanın bizim koşulumuza taşınmamasıydı — yani zaten birinci
> bulgumuz.
>
> İkincisini de ölçümle düzelttik.
>
> **[DUR]**
>
> Ölçüm yapan bir çalışmanın en zayıf yeri, düzeltmediği hatalardır."

---

# ⏱ 6:00 – 6:30 · Kapanış

`▸ Yap:` Sayfa 7'de kal. Konuşma bitince **iki saniye bekle**, sonra kaydı kes.

> "Toparlayayım. Üç cümle.
>
> **Bir:** uç cihaz için model seçimi, yayınlanmış doğruluğa bakarak yapılamaz.
>
> **İki:** doğrulukta ilk grup birbirinden ayrışmıyor; o yüzden seçim enerji ve
> dayanıklılığa kalıyor.
>
> **Üç:** sistem maliyetini modelin boyutu değil, çıktı stride'ı belirliyor.
>
> **[DUR]**
>
> Başta bir kasa düşmüştü, hatırlarsanız. Ölçtük — sistem onu fark etmiyor.
>
> Bu tez o sorunu **çözmüyor.** Ama nerede olduğunu, ne kadar büyük olduğunu ve **neden**
> olduğunu ölçüyor.
>
> Çünkü bir sistemin neyi bilmediğini bilmek, bildiğini sanmaktan iyidir.
>
> **[DUR]**
>
> Panelde gördüğünüz her sayı, diskteki bir ölçüm kaydından okunuyor. Panel canlı çıkarım
> yapmıyor, hiçbir sayı elle yazılmadı.
>
> Teşekkür ederim."

---

# Kayıt notları

## Çekimden hemen önce

- [ ] Tarayıcı **tam ekran** (F11), sol menü görünür
- [ ] Sayfa **1**'de bekliyor
- [ ] Telefon sessiz, masaüstü bildirimleri kapalı
- [ ] Mikrofon testi: bir cümle söyle, geri dinle — tıslama, uğultu, yankı var mı
- [ ] Bu metin **ikinci ekranda** ya da yazdırılmış

## Konuşurken

- **Sayfa geçtikten sonra bir saniye sus.** İzleyicinin gözü yeni sayfaya otursun, sonra
  konuş. En sık atlanan şey bu — ve kaydı en çok amatör gösteren şey de bu.
- **İmleci sürekli oynatma.** Bir şeyi gösterirken götür, orada **bırak**, konuş, sonra
  çek. Titreyen imleç izleyiciyi yorar.
- **Tabloları okuma.** Tablo arka plan, sen ön plansın. Sayfa başına **bir** sayıyı sesli
  söyle, gerisi ekranda dursun.
- **Hızlanma refleksine dikkat.** Kayıtta herkes hızlanır. Bilerek yavaş konuş — fazladan
  yarım dakika payın var.
- **[DUR]** işaretlerini atlama. Metnin ritmi onlara göre kuruldu; atlarsan hem hızlanır
  hem de vurgular kaybolur.

## Bir şeyler ters giderse

| durum | ne yap |
|---|---|
| Cümleyi bozdun | Dur, **üç saniye** bekle, cümleyi baştan al. Boşluk bırakmak kesmeyi kolaylaştırır. |
| Panel takıldı | F5 ile yenile, kaldığın sayfaya dön, o bölümü tekrar al. |
| Süre aşıyor | **§5'i tamamen atla.** Yetmezse §3'teki skor tablosu cümlesini kes, doğrudan `lost` satırına geç. |
| Süre kısa kaldı | Sayfa 6'daki "Koşu" menüsünden başka model seç, eğrilerin yenilendiğini göster: *"her model için ayrı telemetri kaydı var."* On saniye. |

## Vaktin kalırsa söyleyebileceklerin

Kayıt altı dakikanın altında bittiyse aşağıdakilerden **bir** tanesini ekleyebilirsin.
Hepsini sıkıştırma.

*(Not: değerlendirme formunda "soruları yanıtlama yetkinliği" diye 7 puanlık bir satır var.
Dersin bu dönem nasıl işlediğini sen biliyorsun; sözlü bir kısım çıkarsa bunlar hazır
cevap olarak durur.)*

**Kendi modelinizi eğitmediniz mi?**
> Eğittik — Cityscapes ve IDD20K karışımıyla, sınırlı bir bütçede. Ama o modeller
> yayınlanmış checkpoint'lerin yüzde 0,7'si kadar örnek gördü. O yüzden karşılaştırma
> temeli olarak resmî referansları kullandık ve ikisini birbirine **karıştırmadık** —
> tezde ikisi de ayrı bölümlerde.

**n = 5 ile korelasyon konuşulur mu?**
> Konuşulmaz, ve tezde de öyle yazıyor. Nitekim o eksi 0,90 bulgusunu tam bu yüzden geri
> çektik. Sayısal iddialarımız korelasyondan değil, **eşleştirilmiş karşılaştırmadan**
> geliyor — aynı kareler, güven aralığıyla.

**FP16'ya çevirince doğruluk kaybettiniz mi?**
> Ölçtük. Cihazda, FP16 motoruyla FP32'yi **aynı koşuda aynı karelerde** puanladık: fark
> sıfırdan ayırt edilemiyor, piksel uyumu yüzde 99,97. Eşleştirmeseydik yanlış sonuca
> varacaktık — ayrı ölçülmüş sayılarla kıyaslayınca sahte bir ceza çıkıyordu.

**12 FPS gerçek zamanlı sayılır mı?**
> Sayılmaz, ve demiyoruz da. Sınırlar sayfasında "geçilmedi" yazıyor. Ölçtüğümüz şey
> darboğazın nerede olduğu: motorun payı yüzde beş, gerisi CPU tarafı.

**Bu bir ürün mü?**
> Hayır, araştırma prototipi. ISO 26262 kapsamında hiçbir güvenlik seviyesi hedeflenmedi
> ve tezde bu açıkça yazılı. Ama bulgular ISO 21448 — yani SOTIF — sınıfında: sistem
> arızalanmadan, sadece algı yetersiz kaldığı için ortaya çıkan tehlikeler.
