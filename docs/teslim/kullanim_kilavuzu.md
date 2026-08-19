# EdgeGuard-Road — Kullanım Kılavuzu

Bu belge, teslim paketindeki uygulamaların nasıl kullanılacağını anlatır. Kurulum
adımları ayrı bir belgededir: `Kurulum_Talimatlari.pdf`.

## Sistem ne yapar

EdgeGuard-Road, bir yol sahnesi görüntüsünü alıp dört soruyu birlikte yanıtlayan bir algı
hattıdır:

1. **Ne görüyorum?** — 19 sınıflı semantik bölütleme.
2. **Ne kadar eminim?** — piksel başına güven ve entropi haritaları.
3. **Daha önce hiç görmediğim bir şey var mı?** — open-set (açık küme) uyarısı.
4. **Hangi bölge operasyonel olarak önemli?** — bağlamsal risk sıralaması.

Sistem **doğrudan araç kontrolü yapan bir ADAS değildir.** Bir algı prototipidir ve
çıktısı karar destek katmanıdır.

---

## 1. Sunum paneli

Panel, projenin bütün ölçüm sonuçlarını gezilebilir hâlde sunar. Projenin ikinci dönem
sunumu bu panel üzerinden yapılmıştır.

Başlatma komutu `Kurulum_Talimatlari.pdf` belgesindedir.

### Panelin çalışma ilkesi

Panel **canlı çıkarım yapmaz.** Gösterdiği her sayıyı diskteki bir ölçüm kaydından okur.
Bunun nedeni bilinçlidir: bir sunum sırasında modelin yavaş yüklenmesi ya da bir kareye
takılması, anlatılan konuyla ilgisi olmayan bir riske girmek olurdu.

Tek istisna cihazın kendi anlık durumudur. Panel bir Jetson üzerinde çalışıyorsa kenar
çubuğunda o andaki sıcaklık, güç ve bellek değerlerini gösterir. Bu, raporlanan telemetri
**değildir**; raporlanan telemetri, sonuçlar üretilirken kaydedilmiş 600 saniyelik zaman
serileridir. Panel bu ayrımı kendi üzerinde de yazar.

Panel Jetson dışında bir makinede çalıştırıldığında telemetri satırı "okunamıyor" der;
bu bir hata değildir.

### Sayfalar

| Sayfa | İçerik |
|---|---|
| 1 · Problem ve yöntem | Çalışmanın sorusu, dört ölçüm ekseni, kullanılan veri ve modeller |
| 2 · Model karşılaştırması | Yayımlanmış ve ölçülen doğruluk, eşleştirilmiş istatistiksel karşılaştırma, sınıf bazlı IoU |
| 3 · Açık küme yol tehlikesi | Dört belirsizlik skorunun ayırt etme gücü, tehlike türüne göre kırılım |
| 4 · Belirsizlik ve kalibrasyon | Güvenilirlik diyagramları, havuzlanmış ve sınıf bazlı kalibrasyon hatası, aşırı güven |
| 5 · Bağlamsal risk | Sürülebilir alan, bileşen konumlandırma, zamansal davranış, nitel görseller |
| 6 · Uç cihaz: koşu telemetrisi | Sürdürülen yük altında güç, sıcaklık ve bellek eğrileri; gecikme ve enerji çizelgeleri; demo videosu |
| 7 · Sınırlar | Ölçülmeyenler, karşılanmayan hedefler ve geri çekilen iki sonuç |

Sayfalar arasında kenar çubuğundaki menüden geçilir.

### Panelde dikkat edilecek noktalar

- **"Doğrulukta birinci" diye bir model yoktur.** İkinci sayfadaki eşleştirilmiş
  karşılaştırma, ilk iki modelin istatistiksel olarak ayırt edilemediğini gösterir.
  Panel bunu "berabere" olarak yazar; bu bir eksiklik değil, ölçümün sonucudur.
- **Yedinci sayfa kasıtlı olarak vardır.** Ölçülmeyen şeyler ve geri çekilen sonuçlar
  orada listelenir. Bir sonucun geri çekilmiş olması, o sonucun silindiği anlamına
  gelmez.

---

## 2. Görüntü demo uygulaması

`app.py`, tek bir görüntü üzerinde hattın tamamını çalıştıran ayrı bir Streamlit
uygulamasıdır.

```
streamlit run app.py
```

Panelden farklı olarak bu uygulama **gerçek çıkarım yapar** ve bu nedenle model
ağırlıklarına ihtiyaç duyar. Model ağırlıkları lisans nedeniyle teslim paketinde
**bulunmaz** (bkz. `Lisans_Bilgileri.txt`); uygulamayı çalıştırmak için ağırlıkların
resmî kaynağından ayrıca edinilmesi gerekir.

Uygulama, yüklenen görüntü için semantik maskeyi, güven ve entropi haritalarını,
sürülebilir koridoru, bağlantılı bölgeleri ve operasyonel dikkat haritasını üretir.

---

## 3. Sistem çalışma videosu

`Uygulama/Demo_Videosu/EdgeGuard_Sistem_Calisma_Videosu.mp4`

Hareketli bir sahnede sistemin çıktısını gösterir: Cityscapes demo dizisinden 180 ardışık
kare, önceden üretilmiştir. Video, kare kare semantik maskeyi, belirsizlik haritasını ve
risk sıralamasını birlikte gösterir.

Video herhangi bir oynatıcıyla açılabilir; kurulum gerektirmez.

---

## 4. Ölçüm kayıtlarının doğrudan okunması

Panel kullanmadan sonuçlara bakmak isterseniz, ölçüm kayıtları makine tarafından
okunabilir JSON biçimindedir:

`Uygulama/Panel_Sonuc_Paketi/` altında 56 kayıt bulunur.

Her kayıt şu alanları taşır:

| Alan | Anlamı |
|---|---|
| `schema_version` | Kayıt biçiminin sürümü |
| `record_type` | Ölçümün türü |
| `generated_at` | UTC üretim zamanı |
| `model_sha256` | Ölçülen ağırlık dosyasının özeti |
| `frames` | Kullanılan kare sayısı |
| `seed` | Rastgelelik tohumu |
| `scientific_status` | Ölçümün gerçekten çalıştırılıp çalıştırılmadığı |

`model_sha256` alanı sayesinde bir sayının hangi ağırlıkla üretildiği kayıttan
doğrulanabilir. `scientific_status` alanı `not_run` / `measured` / `accepted`
değerlerini alır ve **her zaman gerçekte ne çalıştırıldığını gösterir**, beklenen ya da
umulan değeri değil.

---

## 5. Raporda geçen bir sayıyı doğrulama

Tezde geçen dört ondalıklı her sayının bir ölçüm kaydında karşılığı olduğu otomatik
olarak denetlenmiştir. Bunu kendiniz de çalıştırabilirsiniz:

```
python scripts/verify_reported_numbers.py
```

Betik, rapor metnindeki sayıları tarar ve ölçüm kayıtlarıyla eşleştirir; eşleşmeyen bir
sayı bulursa bildirir. Kayıtlarda doğrudan bulunmayan türetilmiş değerler (iki kaydın
farkı gibi) ayrı bir listede, türetiliş biçimiyle birlikte belirtilir.

---

## Bilinen sınırlar

Bunlar gizlenmemiştir; tezin 5.12 bölümünde de yazılıdır.

- **Gerçek zaman hedefi karşılanmamıştır.** En iyi sonuç 12,36 FPS, hedef 20 idi.
  Darboğazın model değil CPU tarafındaki algı yığını olduğu ölçülmüştür.
- **Sistem yola düşmüş yükü ayırt edememektedir** (AUROC 0,4805 — rastgeleden kötü).
  Bu, projenin güvenlik açısından en önemli bulgusudur ve saklanmamıştır.
- **Gece koşulunda bütün mimariler doğruluğunun en az %70'ini kaybeder** ve kalibrasyon
  hatası 8,7 ile 14,0 kat artar.
- **Mühürlü nihai test verisi açılmamıştır.** Bu kasıtlı bir kısıttır.
