# Değişiklik raporu — EdgeGuard-Road / BM498

Üretilen dosya: **`EdgeGuard_BM498_tez.docx`** (67 sayfa; ön kısım i–xv, gövde 1–52).
Yanında `EdgeGuard_BM498_tez_onizleme.pdf` bir LibreOffice çıktısıdır — yalnızca
yerleşimi görmek içindir, alan değerleri için değil (aşağıda "Bilinen görüntüleme
sınırı" başlığına bakınız).

---

## 1 · İlk açtığınızda yapmanız gereken tek şey

Word'de açın, **Ctrl+A → F9** (veya Baskı Önizleme). Bu, içindekiler tablosunu,
şekil/çizelge listelerini, şekil ve çizelge numaralarını ve metin içi çapraz
başvuruları günceller. Sayfa numaraları zaten doğru yazılıdır; F9 onları
yeniden doğrular.

---

## 2 · Değiştirilen numaralar — dikkatle okuyun

Biçim kuralları şekil ve çizelge numaralarının **Word'ün otomatik alanlarından**
gelmesini şart koşuyor (`STYLEREF 1 \s` + `SEQ`). Word bu alanları **belge
sırasına göre** sayar. Kaynak metinde üç numara belge sırasıyla uyuşmuyordu;
otomatik numaralandırma bunları yeniden atadı.

| Dosya / kaynak numara | Yeni numara | Neden |
|---|---|---|
| `Sekil_5.7_asiri_guven.pdf` (Şekil 5.7) | **Şekil 5.4** | Bu şekil §5.4'te, `Sekil_5.5_open_set` (§5.6) ve `Sekil_5.6_pareto`'dan (§5.7) **önce** geçiyor. |
| `Sekil_5.4_acdc_kosullari.pdf` (Şekil 5.4) | **Şekil 5.7** | Bu şekil §5.8'de, yukarıdakilerden **sonra** geçiyor. |
| Çizelge 10.1 (EK 1) | **Çizelge 8.1** | Bu tezde EKLER 8. bölümdür (şablonun örneğinde 10'du). |

**Yerleşimler değişmedi.** Her şekil, `PROMPT.md`'deki eşleme tablosunun
belirttiği bölümdedir. Diğer 14 şeklin numarası aynı kaldı. Metin içi bütün
atıflar çapraz başvuru (REF alanı) olduğu için kendiliğinden doğru numarayı
gösterir — elle düzeltmeniz gereken bir yer yok.

Sonuç olarak ŞEKİL LİSTESİ artık 1.1 → 5.10 arasında kesintisiz ve sıralıdır;
kaynak metindeki sırayla bırakılsaydı liste 5.3, 5.7, 5.5, 5.6, 5.4 diye
atlayacaktı.

---

## 3 · Eklediklerim

Bunlar bulgu değil, biçim tamamlamasıdır; yine de gözden geçirmenizi öneririm.

### 3.1 · İki çizelge başlığı (EKLER)

EK 1 ve EK 3'te başlıksız duran iki tabloya başlık ekledim; tezdeki diğer 25
tablonun hepsinde başlık vardı.

- `Çizelge 8.2. Ölçüm kaydı şemasının alanları.`
- `Çizelge 8.3. Çalışma zamanı sürüm sabitlemesi.`

### 3.2 · KISALTMALAR ve SİMGELER listeleri — **asıl kontrol etmeniz gereken yer**

Şablonda bu iki sayfa kimya örnekleriyle doluydu (AcO/Asetat, ABS, DEGA;
σ/Öz İletkenlik, Ω/Ohm). Bir bilgisayar mühendisliği tezinde bunların kalması
"göze batan düzensizlik" olurdu, bu yüzden tezin kendi terimleriyle
değiştirdim: 27 kısaltma, 6 simge.

Kısaltmaların kendisi metinden alındı; **açılımlar bana aittir.** Beş dakikada
gözden geçirin, gerekirse kısaltın veya silin.

---

## 4 · Çıkardıklarım

- **HARİTA LİSTESİ sayfası.** Tezde harita yok; sayfa güncellendiğinde
  "girdi bulunamadı" uyarısıyla boş kalacaktı. Bölümünüz bu sayfayı koşulsuz
  istiyorsa geri eklenmelidir.
- **Kapaktan üç boş paragraf.** Tez başlığı bir satır yerine dört satır
  kapladığı için kapak ikinci sayfaya taşıyordu.
- **KISALTMALAR tablosundan sonraki iki boş paragraf.** Araya boş bir sayfa
  giriyordu.
- **ÖZGEÇMİŞ'teki örnek öğrenim satırları** ("Y. Lisans / Elektrik Elektronik
  Müh. / …Üniversitesi / 2012" vb.) — başkasına ait örnek veriydi. Tablo
  iskeleti duruyor.

---

## 5 · Düzelttiğim şablon sorunları

- **Gövdenin sayfa numarası 18'den başlıyordu.** Şablon, arap rakamına geçişi
  (`pgNumType start="1"`) 1. bölümü kapatan paragrafta tutuyor; o paragraf
  şablonun örnek metniyle birlikte gidiyordu. Bölüm sonuna geri taşıdım. Şimdi
  ön kısım i–xv, gövde 1'den başlıyor.
- **ÖZGEÇMİŞ sayfasında numara yoktu.** Şablonun son bölümü romen rakamına
  dönüyor ve boş bir altbilgiye bağlanıyordu. Gövdeyle sürekli olacak şekilde
  düzelttim.
- **Kısaltma sütunu dardı** — "PRISMA" → "PRIS/MA", "AUROC" → "AUR/OC" diye
  bölünüyordu. Sütunu genişlettim.

---

## 6 · Dokunmadıklarım

- **Kapak, değerlendirme ve sözlü sınav tutanağı, BEYAN, ÜRETKEN YAPAY ZEKA
  KULLANIM BEYANI** — tek kelimesi değişmedi. Yalnızca kapaktaki kişisel bilgi
  alanları `[DOLDURULACAK: …]` olarak işaretlendi ve tez başlığı yazıldı.
- **Sayılar.** Kaynak metindeki 971 sayısal belirtecin tamamı, 111 benzersiz
  dört ondalıklı değer dâhil, belgede birebir duruyor (otomatik denetlendi).
  Hiçbir yuvarlama, hiçbir "düzeltme" yapılmadı. `+0,0000` olduğu gibi duruyor.
- **Bulgular ve ifadeler.** "ayırt edilemez", "ölçülmedi", "geri çekildi"
  ifadeleri yerinde. Geri çekilen iki sonuç raporda duruyor. Rapor "en doğru
  model" ifadesini yalnızca *kullanmadığını söylediği* cümlede taşıyor; öyle
  bırakıldı.
- **19 kaynak** — ekleme yok, sıra değişmedi, metin içi atıflar numaralı.

---

## 7 · Doldurmanız gereken yerler

Belgede `[DOLDURULACAK: …]` diye arayın; 14 ayrı yer var.

| Yer | Alan |
|---|---|
| Kapak | akademik yıl · dönem · ders sorumlusu · öğrenci adı · öğrenci numarası |
| TEŞEKKÜR | serbest metin |
| ÖZET | öğrenci adı · danışman · ay-yıl · **sayfa sayısı** |
| ABSTRACT | aynılarının İngilizcesi |
| ÖZGEÇMİŞ | kişisel bilgiler, öğrenim durumu, yayınlar |

Sayfa sayısı için: gövde 52 sayfa, ön kısım xv. Şablonun ÖZET sayfası genellikle
toplam sayfayı ister — kendi bölümünüzün kılavuzuna göre yazın.

**İki beyanı imzalayıp tarihlemeyi unutmayın.** `OKU_BENI.md`'nin belirttiği
gibi, üretken yapay zekâ kullanım beyanını dürüst doldurun; EK 2 zaten bu beyana
dayanak oluşturan kullanım kaydını anlatıyor.

---

## 8 · Bilinen görüntüleme sınırı — önemli

LibreOffice ve Google Docs, şekil/çizelge başlıklarındaki `STYLEREF 1 \s`
alanını desteklemiyor; başlık numarası yerine bölüm **adını** gösteriyorlar:

> Şekil BULGULAR VE TARTIŞMA.13. Gerçek olumsuz koşullarda…

Bunu **şablonun kendisinde de test ettim; şablon da aynı davranıyor.** Yani
alan yapısı doğru, sorun görüntüleyicide. Microsoft Word'de doğru çıkıyor:

> Şekil 5.7. Gerçek olumsuz koşullarda…

**Bu yüzden belgeyi Word'de açıp F9 ile güncelleyin, PDF'i Word'den alın.**
Yanındaki önizleme PDF'i bu sınırı taşır; yerleşim kontrolü için üretildi,
teslim için değil.

---

## 9 · Otomatik olarak doğruladıklarım

| Denetim | Sonuç |
|---|---|
| OOXML şema geçerliliği | geçti |
| Kaynak metindeki sayıların korunması | 971/971 |
| Dört ondalıklı değerler | 111/111 |
| Çapraz başvuru hedefleri (43 REF alanı) | hedefsiz yok |
| İçindekiler / liste bağlantıları (145 PAGEREF) | hedefsiz yok |
| Yer imi başlangıç–bitiş eşleşmesi | 147/147, yineleme yok |
| Sayfa numaraları (iki geçiş arasında kayma) | 0 |
| Sayfa arasında bölünen tablo | 0 |
| Sayfa sonunda tek kalan başlık | 0 |
| Şekil sayısı | 16/16 |
| Çizelge sayısı | 27 (25 kaynak + 2 eklenen) |

---

## 10 · Yine de kendiniz bakmanız gereken şeyler

Otomatik denetim yerleşimi görebilir, ama tercihi göremez:

1. **KISALTMALAR açılımlarını okuyun** (§3.2).
2. Word'de F9'dan sonra **şekil ve çizelge numaralarının** beklediğiniz gibi
   çıktığını bir kez doğrulayın — özellikle §2'deki üç değişikliği.
3. Geniş çizelgeler (5.10 ve 5.12; sekiz sütunlu) 9 punto ile dizildi. Okunur
   ama küçük; bölümünüz alt sınır koyuyorsa yatay sayfaya almak gerekebilir.
4. Baştan sona bir kez göz gezdirin. `OKU_BENI.md`'nin dediği gibi, bunu bana
   güvenerek atlamayın.
